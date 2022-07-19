/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import '@tensorflow-models/face-detection';

import {Camera} from './camera';
import {setupDatGui} from './option_panel';
import {STATE, createDetector} from './shared/params';
import {setupStats} from './shared/stats_panel';
import {setBackendAndEnvFlags} from './shared/util';

let detector;
let camera;
let stats;
let startInferenceTime;
let numInferences = 0;
let inferenceTimeSum = 0;
let lastPanelUpdate = 0;
let rafId;

let headSize = 0.18; // in meters
let headPos = {x: 0, y: 0, z: 0, ang: 0};
let updateInterval = 1000 / 10; // 10 Hz
let sendPayload = false;

const endpointUrl = document.getElementById('jsonrpc');
const linkStatus = document.getElementById('link');
const scaleSlider = document.getElementById('scale');
const scaleValue = document.getElementById('scale-value');
const freqSlider = document.getElementById('freq');
const freqValue = document.getElementById('freq-value');
const payloadText = document.getElementById('payload');
const connectButton = document.getElementById('connect');
const sendButton = document.getElementById('send');

function disconnectPianoteq(className = 'link-inactive') {
  sendPayload = false;
  connectButton.innerText = 'Connect';
  linkStatus.className = className;
  payload.readonly = false;
  sendButton.disabled = false;
}

async function getPianoteqParameters() {
  const url = endpointUrl.value;
  const payload = {
    id: 1,
    jsonrpc: '2.0',
    method: 'getParameters',
    params: {},
  };
  fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  }).then(async (response) => {
    const responseJSON = await response.json();
    const headDiameter = responseJSON.result
      .find((result) => result.id === 'Head Diameter');
    if (headDiameter != null) {
      headSize = Number(headDiameter.text) / 100; // convert to meters
    }
    linkStatus.className = 'link-ok';
  }).catch(() => disconnectPianoteq('link-fail') );
}

async function fireAndForget(payload) {
  const url = endpointUrl.value;
  fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: payload,
  }).then(() => linkStatus.className = 'link-ok')
  .catch(() => disconnectPianoteq('link-fail') );
}

async function sendToPianoTeq() {
  setTimeout(sendToPianoTeq, updateInterval);
  if (!sendPayload) return;
  // console.log(headPos);

  const payloadData = {
    id: 1,
    jsonrpc: '2.0',
    method: 'setParameters',
    params: {
      list: [
        {id: 'Head.X', name: 'Head X position', text: String(headPos.x)},
        {id: 'Head.Y', name: 'Head Y position', text: String(headPos.z)},
        {id: 'Head.Z', name: 'Head Z position', text: String(headPos.y)},
        {id: 'Head Angle', name: 'Head Angle', text: String(headPos.ang)},
      ],
    },
  };

  const payloadJSON = JSON.stringify(payloadData, null, 2);

  payloadText.value = payloadJSON;
  fireAndForget(payloadJSON);
}

async function headTracking(faces) {
  const RIGHT_EAR = 234;
  const LEFT_EAR = 454;

  const kp = faces[0].keypoints;

  const eyesDistance = Math.sqrt(
    Math.pow(kp[LEFT_EAR].x - kp[RIGHT_EAR].x, 2) +
    Math.pow(kp[LEFT_EAR].y - kp[RIGHT_EAR].y, 2) +
    Math.pow(kp[LEFT_EAR].z - kp[RIGHT_EAR].z, 2)
  );
  const scale = Number(scaleSlider.value) * headSize / eyesDistance;

  headPos.ang = Math.atan2(
    kp[LEFT_EAR].z - kp[RIGHT_EAR].z,
    kp[LEFT_EAR].x - kp[RIGHT_EAR].x
  );
  headPos.ang *= 180 / Math.PI;

  headPos.x = camera.video.width / 2;
  headPos.y = camera.video.height / 2;
  headPos.z = 0;
  headPos.x -= kp[LEFT_EAR].x - Math.abs(kp[LEFT_EAR].x - kp[RIGHT_EAR].x) / 2;
  headPos.y -= kp[LEFT_EAR].y - Math.abs(kp[LEFT_EAR].y - kp[RIGHT_EAR].y) / 2;
  headPos.z -= kp[LEFT_EAR].z - Math.abs(kp[LEFT_EAR].z - kp[RIGHT_EAR].z) / 2;
  headPos.x *= scale;
  headPos.y *= scale;
  headPos.z *= scale;

  // PianoTeq default offsets for the head
  headPos.x += 0.620;
  headPos.y += 1.300;
  headPos.z += -0.36;
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimateFaceStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimateFaceStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let faces = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimateFaces.
    beginEstimateFaceStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      faces =
          await detector.estimateFaces(camera.video, {flipHorizontal: false});
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimateFaceStats();
  }

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (faces && faces.length > 0 && !STATE.isModelChanged) {
    headTracking(faces);

    camera.drawResults(
        faces, STATE.modelConfig.triangulateMesh,
        STATE.modelConfig.boundingBox);
  }
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
};

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    urlParams.append('model', 'mediapipe_face_mesh');
  }

  await setupDatGui(urlParams);

  stats = setupStats();

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();

  connectButton.onclick = async () => {
    if (sendPayload) {
      disconnectPianoteq();
    } else {
      sendPayload = true;
      connectButton.innerText = 'Disconnect';
      payload.readonly = true;
      sendButton.disabled = true;
      await getPianoteqParameters();
    }
  };

  sendButton.onclick = async () => {
    fireAndForget(payloadText.value);
  };

  scaleSlider.oninput = () => scaleValue.innerText = `x${scaleSlider.value}`;

  freqSlider.oninput = () => {
    updateInterval = 1000 / Number(freqSlider.value);
    freqValue.innerText = `${freqSlider.value} Hz`;
  };

  setTimeout(sendToPianoTeq, updateInterval);

  renderPrediction();
};

app();

import gaze from "../../index.js";
import { initialLetters } from "./utils.js";

let letters = initialLetters;
const leftSection = document.querySelector(".section-left");
const rightSection = document.querySelector(".section-right");

const initKeyboardUI = (lettersSelected, update) => {
  if (leftSection.hasChildNodes() && update) {
    leftSection.innerHTML = "";
  }
  if (rightSection.hasChildNodes() && update) {
    rightSection.innerHTML = "";
  }
  if (lettersSelected.length === 1) {
    const outputElement = document.querySelector(".output");
    if (lettersSelected[0] === "Delete") {
      outputElement.value = outputElement.value.slice(0, -1);
    } else if (lettersSelected[0] === "space") {
      outputElement.value += " ";
    } else if (lettersSelected[0] === "Enter") {
      outputElement.value = `${outputElement.value}\n`;
    } else {
      outputElement.value += lettersSelected[0];
    }

    initKeyboardUI(initialLetters, false);
    letterPrinted();
    return;
  } else {
    lettersSelected.map((letter, i) => {
      const letterElement = document.createElement("p");
      letterElement.innerHTML = letter;
      if (i < Math.round(lettersSelected.length / 2)) {
        leftSection.appendChild(letterElement);
      } else {
        rightSection.appendChild(letterElement);
      }
    });
  }
};

initKeyboardUI(letters, false);

const letterPrinted = () => (letters = initialLetters);

let previousGazeDirection;

const updateModelStatus = () => {
  const status = document.querySelector(".model-status");
  if (status) {
    status.innerHTML = "Model loaded! You can start!";
    status.classList.add("fade-out");
    status.classList.remove("model-status");
  }
};

const videoElement = document.querySelector("video");

const init = async () => {
  await gaze.loadModel();
  await gaze.setUpCamera(videoElement);

  const predict = async () => {
    let direction = await gaze.getGazePrediction();
    updateModelStatus();

    if (direction === "RIGHT") {
      if (previousGazeDirection !== direction) {
        letters = letters.slice(Math.round(letters.length / 2), letters.length);
        initKeyboardUI(letters, true);
        previousGazeDirection = direction;
      }
    } else if (direction === "LEFT") {
      if (previousGazeDirection !== direction) {
        letters = letters.slice(0, Math.round(letters.length / 2));
        initKeyboardUI(letters.slice(letters), true);
        previousGazeDirection = direction;
      }
    } else if (direction === "STRAIGHT") {
      previousGazeDirection = direction;
    }

    let raf = requestAnimationFrame(predict);
  };
  predict();
};

init();

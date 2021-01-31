export const initialLetters = [
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
  "Delete",
  "space",
  "Enter",
];

export const commands = ["Delete", "Space", "Enter"];

export const VIDEO_SIZE = 500;

const isMobile = () => {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
};

export const state = {
  backend: "webgl",
  maxFaces: 1,
  triangulateMesh: true,
  predictIrises: true,
};

export const mobile = isMobile();

export const normalize = (val, max, min) =>
  Math.max(0, Math.min(1, (val - min) / (max - min)));

export const isEven = (value) => {
  if (value % 2 == 0) return true;
  else return false;
};

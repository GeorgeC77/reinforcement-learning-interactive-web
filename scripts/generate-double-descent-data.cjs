const fs = require('fs');
const path = require('path');

/* -------------------------------------------------------------------------- */
/* 数值工具（与页面端保持一致）                                               */
/* -------------------------------------------------------------------------- */
function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
    }
    [M[i], M[maxRow]] = [M[maxRow], M[i]];

    const pivot = M[i][i];
    if (Math.abs(pivot) < 1e-12) continue;

    for (let j = i; j <= n; j++) M[i][j] /= pivot;

    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const factor = M[k][i];
      for (let j = i; j <= n; j++) M[k][j] -= factor * M[i][j];
    }
  }

  return M.map((row) => row[n]);
}

function matMul(A, B) {
  return A.map((row) => B[0].map((_, j) => row.reduce((sum, v, k) => sum + v * B[k][j], 0)));
}

function transpose(A) {
  return A[0].map((_, j) => A.map((row) => row[j]));
}

function generateGaussianMatrix(rows, cols, seed) {
  let s = seed;
  const M = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      s = (s * 9301 + 49297) % 233280;
      const u = Math.max(1e-10, s / 233280);
      s = (s * 9301 + 49297) % 233280;
      const v = s / 233280;
      row.push(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
    }
    M.push(row);
  }
  return M;
}

function generateDataLinear(n, d, noiseStd, seed) {
  const betaTrue = Array.from({ length: d }, (_, j) => (j < 5 ? 1.0 : 0.0));
  const X = generateGaussianMatrix(n, d, seed);
  let s = seed + 7;
  const y = X.map((row) => {
    const pred = row.reduce((sum, xj, j) => sum + xj * betaTrue[j], 0);
    s = (s * 9301 + 49297) % 233280;
    const u = Math.max(1e-10, s / 233280);
    s = (s * 9301 + 49297) % 233280;
    const v = s / 233280;
    return pred + noiseStd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  });
  return { X, y, betaTrue };
}

function fitLinearModel(X, y) {
  const n = X.length;
  const d = X[0].length;

  if (d <= n) {
    const Xt = transpose(X);
    const XtX = matMul(Xt, X);
    const Xty = Xt.map((row) => row.reduce((sum, v, i) => sum + v * y[i], 0));
    return solveLinearSystem(XtX, Xty);
  } else {
    const Xt = transpose(X);
    const XXt = matMul(X, Xt);
    const alpha = solveLinearSystem(XXt, y);
    return Xt.map((row) => row.reduce((sum, v, i) => sum + v * alpha[i], 0));
  }
}

function mseLinear(X, beta, y) {
  const pred = X.map((row) => row.reduce((sum, xj, j) => sum + xj * beta[j], 0));
  return pred.reduce((sum, p, i) => sum + Math.pow(p - y[i], 2), 0) / pred.length;
}

/* -------------------------------------------------------------------------- */
/* 生成双下降曲线数据                                                         */
/* -------------------------------------------------------------------------- */
function generateCurve(n, noise, maxD, numTrials, seedOffset) {
  const result = [];
  for (let d = 1; d <= maxD; d++) {
    let trainSum = 0;
    let testSum = 0;
    let validTrials = 0;
    for (let t = 0; t < numTrials; t++) {
      try {
        const train = generateDataLinear(n, d, noise, seedOffset * 100000 + d * 1000 + t);
        const test = generateDataLinear(200, d, noise, seedOffset * 100000 + d * 1000 + t + 50000);
        const betaHat = fitLinearModel(train.X, train.y);
        const trainErr = mseLinear(train.X, betaHat, train.y);
        const testErr = mseLinear(test.X, betaHat, test.y);
        if (Number.isFinite(trainErr) && Number.isFinite(testErr)) {
          trainSum += trainErr;
          testSum += testErr;
          validTrials++;
        }
      } catch {
        // 数值不稳定时跳过
      }
    }
    if (validTrials > 0) {
      result.push({ d, train: trainSum / validTrials, test: testSum / validTrials });
    }
  }
  return result;
}

const PARAMS = {
  nValues: [20, 40, 60],
  noiseValues: [0.1, 0.3, 0.5],
  maxDValues: [80, 120, 160],
  numTrials: 15,
  seedOffset: 0,
};

const dataset = [];
let completed = 0;
const total = PARAMS.nValues.length * PARAMS.noiseValues.length * PARAMS.maxDValues.length;

for (const n of PARAMS.nValues) {
  for (const noise of PARAMS.noiseValues) {
    for (const maxD of PARAMS.maxDValues) {
      console.log(`[${++completed}/${total}] Computing n=${n}, noise=${noise}, maxD=${maxD}`);
      const curve = generateCurve(n, noise, maxD, PARAMS.numTrials, PARAMS.seedOffset);
      dataset.push({ n, noise, maxD, numTrials: PARAMS.numTrials, curve });
    }
  }
}

const outDir = path.join(__dirname, '..', 'public', 'data');
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}
const outPath = path.join(outDir, 'double-descent-curves.json');
fs.writeFileSync(outPath, JSON.stringify({ params: PARAMS, dataset }, null, 2));
console.log(`Wrote ${outPath} (${dataset.length} curves)`);

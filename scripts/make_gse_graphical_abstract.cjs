"use strict";

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const root = path.resolve(__dirname, "..");
const output = path.join(root, "paper", "gse", "figs", "graphical_abstract.png");
const tiffOutput = path.join(root, "paper", "gse", "figs", "graphical_abstract.tiff");

const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="3120" height="1050" viewBox="0 0 3120 1050">
  <rect width="3120" height="1050" fill="#ffffff"/>
  <defs>
    <marker id="arrow" markerWidth="16" markerHeight="16" refX="12" refY="5" orient="auto">
      <path d="M0,0 L12,5 L0,10 Z" fill="#5f6b73"/>
    </marker>
  </defs>
  <style>
    text { font-family: Arial, Helvetica, sans-serif; fill: #27343a; }
    .head { font-size: 54px; font-weight: 700; }
    .title { font-size: 45px; font-weight: 700; }
    .body { font-size: 35px; }
    .metric { font-size: 39px; font-weight: 700; }
  </style>

  <text x="520" y="100" text-anchor="middle" class="head" fill="#c66e43">Optimistic benchmark</text>
  <text x="1560" y="100" text-anchor="middle" class="head" fill="#244b5a">Validation framework</text>
  <text x="2600" y="100" text-anchor="middle" class="head" fill="#3e7c68">Deployment evidence</text>

  <rect x="80" y="180" width="880" height="620" rx="36" fill="#f7e5da" stroke="#c66e43" stroke-width="8"/>
  <text x="520" y="300" text-anchor="middle" class="title" fill="#c66e43">Within-well depth split</text>
  <text x="520" y="430" text-anchor="middle" class="body">
    <tspan x="520" dy="0">Neighbouring depths occur in</tspan>
    <tspan x="520" dy="55">both training and test sets</tspan>
  </text>
  <text x="520" y="650" text-anchor="middle" class="metric">Macro-F1 = 0.737</text>

  <rect x="1120" y="180" width="880" height="620" rx="36" fill="#ddeaf2" stroke="#244b5a" stroke-width="8"/>
  <text x="1560" y="300" text-anchor="middle" class="title" fill="#244b5a">Hold out complete wells</text>
  <text x="1560" y="420" text-anchor="middle" class="body">
    <tspan x="1560" dy="0">State the target-well information budget</tspan>
    <tspan x="1560" dy="70">Report mean and coverage-aware fixed tails</tspan>
    <tspan x="1560" dy="70">Repeat across locked well splits</tspan>
  </text>

  <rect x="2160" y="180" width="880" height="620" rx="36" fill="#e3f0eb" stroke="#3e7c68" stroke-width="8"/>
  <text x="2600" y="300" text-anchor="middle" class="title" fill="#3e7c68">Cross-well results</text>
  <text x="2600" y="420" text-anchor="middle" class="body">
    <tspan x="2600" dy="0">Inductive Macro-F1 = 0.339</tspan>
    <tspan x="2600" dy="70">Post-drilling = 0.370–0.379</tspan>
    <tspan x="2600" dy="70">Fixed-tail scores may not improve</tspan>
  </text>

  <line x1="975" y1="490" x2="1095" y2="490" stroke="#5f6b73" stroke-width="12" marker-end="url(#arrow)"/>
  <line x1="2015" y1="490" x2="2135" y2="490" stroke="#5f6b73" stroke-width="12" marker-end="url(#arrow)"/>

  <text x="1560" y="925" text-anchor="middle" font-size="39" font-weight="700">
    Rare facies remain unreliable: use the framework for reporting completeness and interpreter review.
  </text>
</svg>`;

fs.mkdirSync(path.dirname(output), { recursive: true });
Promise.all([
  sharp(Buffer.from(svg)).png().toFile(output),
  sharp(Buffer.from(svg)).withMetadata({ density: 300 }).tiff({ compression: "lzw" }).toFile(tiffOutput),
]).then(() => {
  console.log(output);
  console.log(tiffOutput);
});

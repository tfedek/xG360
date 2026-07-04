const { Document, Packer } = require('docx');
const fs = require('fs');
const F = require('./formulas');
const { p, h2 } = require('./helpers');

const doc = new Document({
  sections: [{
    children: [
      h2('Test formula'),
      p('Test udaljenosti:'),
      F.eqDistance(),
      p('Test ugla:'),
      F.eqAngle(),
      p('Test sigmoida:'),
      F.eqSigmoidFixed(),
      p('Test VIF:'),
      F.eqVIF(),
      p('Test Brier:'),
      F.eqBrier(),
      p('Test SHAP:'),
      F.eqShap(),
      p('Test chi-square:'),
      F.eqChiSquare(),
      p('Test Odds Ratio:'),
      F.eqOddsRatio(),
      p('Test logit:'),
      F.eqLogit(),
      p('Test pressure score:'),
      F.eqPressureScore(),
      p('Test point-to-segment:'),
      F.eqPointToSegment(),
      p('Test open angle ratio:'),
      F.eqOpenAngleRatio(),
      p('Test Box-Tidwell:'),
      F.eqBoxTidwell(),
      p('Test Mann-Whitney:'),
      F.eqMannWhitney(),
      p('Test KS:'),
      F.eqKS(),
      p('Test GK anomaly:'),
      F.eqGkAnomaly(),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync('/home/claude/docx_build/test_formulas.docx', buf);
  console.log('OK');
});

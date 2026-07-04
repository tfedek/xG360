const {
  Paragraph, TextRun, AlignmentType,
  Math: M, MathRun, MathFraction, MathRadical, MathSum, MathSubScript,
  MathSuperScript, MathSubSuperScript, MathRoundBrackets, MathFunction,
  MathFunctionName,
} = require('docx');

function eq(mathChildren, eqNumber) {
  const children = [new M({ children: mathChildren })];
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 200 },
    tabStops: eqNumber ? [{ type: 'right', position: 9000 }] : undefined,
    children: eqNumber
      ? [...children, new TextRun({ text: `\t(${eqNumber})`, size: 20 })]
      : children,
  });
}

function r(text) {
  return new MathRun(text);
}

function eqDistance() {
  return eq([
    r('d = '),
    new MathRadical({
      children: [
        new MathSuperScript({ children: [new MathRoundBrackets({ children: [r('120 - x')] })], superScript: [r('2')] }),
        r(' + '),
        new MathSuperScript({ children: [new MathRoundBrackets({ children: [r('40 - y')] })], superScript: [r('2')] }),
      ],
    }),
  ], '1');
}

function eqAngle() {
  return eq([
    r('θ = arccos'),
    new MathRoundBrackets({
      children: [
        new MathFraction({
          numerator: [r('a² + b² - c²')],
          denominator: [r('2ab')],
        }),
      ],
    }),
  ], '2');
}

function eqOpenAngleRatio() {
  return eq([
    r('r = 1 - '),
    new MathFraction({
      numerator: [r('min(θ_blok, θ)')],
      denominator: [r('θ')],
    }),
  ], '3');
}

function eqPointToSegment() {
  return eq([
    r('t = '),
    new MathFraction({
      numerator: [r('(P - A) · (B - A)')],
      denominator: [r('|B - A|²')],
    }),
    r(',  t ∈ [0, 1]'),
  ], '4');
}

function eqPressureScore() {
  return eq([
    r('pressure = '),
    new MathSum({
      subScript: [r('dᵢ ≤ 10')],
      superScript: [r(' ')],
      children: [r('1 / (dᵢ + 0,5)')],
    }),
  ], '5');
}

function eqLogit() {
  return eq([
    r('logit(p) = ln'),
    new MathRoundBrackets({
      children: [new MathFraction({ numerator: [r('p')], denominator: [r('1 - p')] })],
    }),
    r(' = '),
    new MathSubScript({ children: [r('β')], subScript: [r('0')] }),
    r(' + '),
    new MathSubScript({ children: [r('β')], subScript: [r('1')] }),
    new MathSubScript({ children: [r('x')], subScript: [r('1')] }),
    r(' + ⋯ + '),
    new MathSubScript({ children: [r('β')], subScript: [r('k')] }),
    new MathSubScript({ children: [r('x')], subScript: [r('k')] }),
  ], '6');
}

function eqSigmoidFixed() {
  return eq([
    r('p = '),
    new MathFraction({
      numerator: [r('1')],
      denominator: [
        r('1 + '),
        new MathSuperScript({ children: [r('e')], superScript: [r('-z')] }),
      ],
    }),
    r(',   z = '),
    new MathSubScript({ children: [r('β')], subScript: [r('0')] }),
    r(' + '),
    new MathSum({
      subScript: [r('i=1')],
      superScript: [r('k')],
      children: [
        new MathSubScript({ children: [r('β')], subScript: [r('i')] }),
        new MathSubScript({ children: [r('x')], subScript: [r('i')] }),
      ],
    }),
  ], '7');
}

function eqOddsRatio() {
  return eq([
    r('OR = '),
    new MathSuperScript({
      children: [r('e')],
      superScript: [new MathSubScript({ children: [r('β')], subScript: [r('i')] })],
    }),
  ], '8');
}

function eqVIF() {
  return eq([
    new MathSubScript({ children: [r('VIF')], subScript: [r('i')] }),
    r(' = '),
    new MathFraction({
      numerator: [r('1')],
      denominator: [
        r('1 - '),
        new MathSuperScript({
          children: [new MathSubScript({ children: [r('R')], subScript: [r('i')] })],
          superScript: [r('2')],
        }),
      ],
    }),
  ], '9');
}

function eqBoxTidwell() {
  return eq([
    r('logit(p) = β₀ + β₁x + β₂ x·ln(x)'),
  ], '10');
}

function eqBrier() {
  return eq([
    r('BS = '),
    new MathFraction({ numerator: [r('1')], denominator: [r('N')] }),
    new MathSum({
      subScript: [r('i=1')],
      superScript: [r('N')],
      children: [
        new MathSuperScript({
          children: [new MathRoundBrackets({ children: [r('pᵢ - oᵢ')] })],
          superScript: [r('2')],
        }),
      ],
    }),
  ], '11');
}

function eqShap() {
  return eq([
    r('f(x) = φ₀ + '),
    new MathSum({
      subScript: [r('i=1')],
      superScript: [r('k')],
      children: [new MathSubScript({ children: [r('φ')], subScript: [r('i')] })],
    }),
  ], '12');
}

function eqChiSquare() {
  return eq([
    r('χ² = '),
    new MathSum({
      subScript: [r('i=1')],
      superScript: [r('n')],
      children: [
        new MathFraction({
          numerator: [new MathSuperScript({ children: [new MathRoundBrackets({ children: [r('Oᵢ - Eᵢ')] })], superScript: [r('2')] })],
          denominator: [r('Eᵢ')],
        }),
      ],
    }),
  ], '13');
}

function eqMannWhitney() {
  return eq([
    r('U = n₁n₂ + '),
    new MathFraction({ numerator: [r('n₁(n₁ + 1)')], denominator: [r('2')] }),
    r(' - R₁'),
  ], '14');
}

function eqKS() {
  return eq([
    r('D = '),
    new MathFunction({
      name: [r('sup')],
      children: [r('|F₁(x) - F₂(x)|')],
    }),
  ], '15');
}

function eqLikelihoodRatio() {
  return eq([
    r('LR = 2'),
    new MathRoundBrackets({
      children: [
        new MathSubScript({ children: [r('ℓ')], subScript: [r('B')] }),
        r(' − '),
        new MathSubScript({ children: [r('ℓ')], subScript: [r('A')] }),
      ],
    }),
    r(' ~ χ²(df)'),
  ], '17');
}

function eqGkAnomaly() {
  return eq([
    r('anomaly = '),
    new MathSubScript({ children: [r('d')], subScript: [r('gk')] }),
    r(' − (a·'),
    new MathSubScript({ children: [r('d')], subScript: [r('shot')] }),
    r(' + b)'),
  ], '16');
}

module.exports = {
  eq, r,
  eqDistance, eqAngle, eqOpenAngleRatio, eqPointToSegment, eqPressureScore,
  eqLogit, eqSigmoidFixed, eqOddsRatio, eqVIF, eqBoxTidwell, eqBrier,
  eqShap, eqChiSquare, eqMannWhitney, eqKS, eqGkAnomaly, eqLikelihoodRatio,
};

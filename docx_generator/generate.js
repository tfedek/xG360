const { Document, Packer, LevelFormat, AlignmentType } = require('docx');
const fs = require('fs');

const { titlePage, DOC_TITLE } = require('./part1_title');
const { buildPart2 } = require('./part2_intro');
const { buildPart3 } = require('./part3_methodology');
const { buildPart4 } = require('./part4_results');
const { buildPart5 } = require('./part5_discussion');
const { pageProps, makeHeader, makeFooter } = require('./helpers');

const bodyChildren = [
  ...buildPart2(),
  ...buildPart3(),
  ...buildPart4(),
  ...buildPart5(),
];

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: 'Calibri', size: 22 } },
    },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: 'Calibri', color: '1F3864' },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: 'Calibri', color: '2E75B6' },
        paragraph: { spacing: { before: 260, after: 140 }, outlineLevel: 1 },
      },
      {
        id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 22, bold: true, italics: true, font: 'Calibri', color: '444444' },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: 'bullets',
        levels: [{ level: 0, format: LevelFormat.BULLET, text: '\u2022', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
      },
      {
        reference: 'numbers',
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
      },
    ],
  },
  sections: [
    titlePage,
    {
      properties: { page: pageProps.page },
      headers: { default: makeHeader(DOC_TITLE) },
      footers: { default: makeFooter() },
      children: bodyChildren,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync('/home/claude/docx_build/xg_rad_nacrt.docx', buffer);
  console.log('Sačuvano: xg_rad_nacrt.docx');
});

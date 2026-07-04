const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak, TabStopType,
  TabStopPosition,
} = require('docx');
const fs = require('fs');

const CONTENT_DXA = 9360; // 1in margine, US Letter
const noBrd = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
const noBrds = { top: noBrd, bottom: noBrd, left: noBrd, right: noBrd };
const tblBrd = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };
const tblBrds = { top: tblBrd, bottom: tblBrd, left: tblBrd, right: tblBrd };

const pageProps = {
  page: {
    size: { width: 12240, height: 15840 },
    margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
  },
};

function empty() {
  return new Paragraph({ children: [new TextRun({ text: '' })] });
}

function makeHeader(text) {
  return new Header({
    children: [
      new Paragraph({
        alignment: AlignmentType.RIGHT,
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF', space: 4 } },
        children: [new TextRun({ text, size: 16, font: 'Calibri', color: '666666', italics: true })],
      }),
    ],
  });
}

function makeFooter() {
  return new Footer({
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: 'Strana ', size: 16, font: 'Calibri', color: '666666' }),
          new TextRun({ children: [PageNumber.CURRENT], size: 16, font: 'Calibri', color: '666666' }),
        ],
      }),
    ],
  });
}

function h1(text, numbered = true, prefix = '') {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text: prefix ? `${prefix}. ${text}` : text })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text })],
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 160, line: 276 },
    indent: opts.noIndent ? undefined : { firstLine: 397 },
    children: [new TextRun({ text, size: 22, font: 'Calibri', ...opts.run })],
  });
}

function pNoJustify(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, size: 22, font: 'Calibri', ...opts.run })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 240 },
    children: [new TextRun({ text, size: 18, font: 'Calibri', italics: true, color: '444444' })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, size: 22, font: 'Calibri' })],
  });
}

function numbered(text) {
  return new Paragraph({
    numbering: { reference: 'numbers', level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, size: 22, font: 'Calibri' })],
  });
}

// Tabela: header red + podaci, kolone se daju kao niz širina (DXA), zbir = CONTENT_DXA
function makeTable(headerCells, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headerCells.map((txt, i) => new TableCell({
      borders: tblBrds,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: '1F3864', type: ShadingType.CLEAR },
      verticalAlign: VerticalAlign.CENTER,
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: txt, bold: true, color: 'FFFFFF', size: 19, font: 'Calibri' })],
      })],
    })),
  });
  const dataRows = rows.map((row, ri) => new TableRow({
    children: row.map((txt, i) => new TableCell({
      borders: tblBrds,
      width: { size: colWidths[i], type: WidthType.DXA },
      verticalAlign: VerticalAlign.CENTER,
      shading: { fill: ri % 2 === 0 ? 'FFFFFF' : 'F2F2F2', type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 120, right: 120 },
      children: [new Paragraph({
        alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: [new TextRun({ text: String(txt), size: 19, font: 'Calibri' })],
      })],
    })),
  }));
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

module.exports = {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak,
  TabStopType, TabStopPosition,
  CONTENT_DXA, noBrd, noBrds, tblBrd, tblBrds, pageProps,
  empty, makeHeader, makeFooter, h1, h2, h3, p, pNoJustify, caption,
  bullet, numbered, makeTable,
};

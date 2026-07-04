const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak,
  CONTENT_DXA, noBrd, noBrds, tblBrds, pageProps,
  empty, makeHeader, makeFooter, h1, h2, h3, p, pNoJustify, caption,
  bullet, numbered, makeTable,
} = require('./helpers');

const DOC_TITLE = 'Doprinos StatsBomb 360 prostornih podataka u proceni verovatnoće gola (xG)';

// ============================================================
// NASLOVNA STRANA
// ============================================================
const titlePage = {
  properties: { page: pageProps.page },
  children: [
    empty(), empty(),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 0, after: 80 },
      children: [new TextRun({ text: 'STATISTIKA I MAŠINSKO UČENJE U ANALIZI SPORTSKIH PODATAKA', bold: true, size: 24, font: 'Calibri', color: '1F3864' })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 0, after: 800 },
      children: [new TextRun({ text: 'Samostalni istraživački projekat', size: 22, font: 'Calibri' })],
    }),
    empty(), empty(),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120 },
      children: [new TextRun({ text: 'Doprinos StatsBomb 360 prostornih podataka u proceni verovatnoće gola (xG)', bold: true, size: 32, font: 'Cambria', color: '2E75B6' })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 0, after: 800 },
      children: [new TextRun({ text: 'Komparativna analiza klasičnih i prostorno-svesnih xG modela', italics: true, size: 24, font: 'Calibri', color: '666666' })],
    }),
    empty(), empty(), empty(), empty(),
    new Table({
      width: { size: CONTENT_DXA, type: WidthType.DXA },
      columnWidths: [CONTENT_DXA / 2, CONTENT_DXA / 2],
      borders: { top: noBrd, bottom: noBrd, left: noBrd, right: noBrd, insideH: noBrd, insideV: noBrd },
      rows: [new TableRow({
        children: [
          new TableCell({
            borders: noBrds, width: { size: CONTENT_DXA / 2, type: WidthType.DXA },
            children: [
              new Paragraph({ children: [new TextRun({ text: 'Autor:', bold: true, size: 22, font: 'Calibri' })] }),
              new Paragraph({ children: [new TextRun({ text: 'Tomislav Fedek', size: 22, font: 'Calibri' })] }),
            ],
          }),
          new TableCell({
            borders: noBrds, width: { size: CONTENT_DXA / 2, type: WidthType.DXA },
            children: [
              new Paragraph({ children: [new TextRun({ text: 'Mentor / konsultacija:', bold: true, size: 22, font: 'Calibri' })] }),
              new Paragraph({ children: [new TextRun({ text: 'prof. dr Zoltán Pap (predlog)', size: 22, font: 'Calibri' })] }),
            ],
          }),
        ],
      })],
    }),
    empty(), empty(), empty(),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: 'Subotica, 2026.', size: 22, font: 'Calibri' })],
    }),
  ],
};

module.exports = { titlePage, DOC_TITLE };

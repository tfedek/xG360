const { Document, Packer, Paragraph, Math: M, MathRun } = require('docx');
const fs = require('fs');

const doc = new Document({
  sections: [{
    children: [
      new Paragraph({
        children: [new M({ children: [new MathRun('x = 5')] })],
      }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync('/home/claude/docx_build/minimal_test.docx', buf);
  console.log('OK');
});

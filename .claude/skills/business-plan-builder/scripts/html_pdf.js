// Print an HTML file to PDF with Chromium. Usage: node html_pdf.js in.html out.pdf
const { chromium } = require("playwright");
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + require("path").resolve(process.argv[2]), { waitUntil: "load" });
  await page.pdf({ path: process.argv[3], preferCSSPageSize: true, printBackground: true });
  await browser.close();
})();

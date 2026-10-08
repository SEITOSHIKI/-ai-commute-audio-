// Render the four logo SVGs to PNG for PowerPoint (SKILL.md §6). Requires: npm i sharp
import sharp from "sharp";
import { readdir } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const dir = join(dirname(fileURLToPath(import.meta.url)), "logos");
for (const f of (await readdir(dir)).filter((n) => n.endsWith(".svg"))) {
  await sharp(join(dir, f), { density: 600 }).png().toFile(join(dir, f.replace(/\.svg$/, ".png")));
  console.log("rendered", f);
}

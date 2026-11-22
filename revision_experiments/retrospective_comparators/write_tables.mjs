import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { Workbook } from '@oai/artifact-tool';
const out = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const replaceExisting=process.argv.includes('--replace-existing');
async function writeReviewed(dest,text) {
  let exists=false;
  try { await fs.readFile(dest,'utf8'); exists=true; } catch(e) { if(e.code !== 'ENOENT') throw e; }
  if(exists && !replaceExisting) throw Error('Existing output preserved: '+dest);
  await fs.writeFile(dest,text,{encoding:'utf8',flag:exists?'w':'wx'});
}

const payload = JSON.parse(await fs.readFile(path.join(out,'analysis_payload.json'),'utf8'));
if (payload.status !== 'complete_historical_descriptive_reanalysis') throw Error('Unverified payload');
const expected = {i3_per_seed:24,i3_summary:4,i7_per_seed:30,i7_summary:5,audited_source_budget_curves:216};
const book = Workbook.create();
const tables = [];
for (const [name,rows] of Object.entries(payload.tables)) {
  if (rows.length !== expected[name]) throw Error('Row count mismatch '+name);
  const cols = Object.keys(rows[0]);
  if (!rows.every(r => JSON.stringify(Object.keys(r)) === JSON.stringify(cols))) throw Error('Schema mismatch');
  const values = [cols,...rows.map(r => cols.map(c => r[c] ?? null))];
  const sheet = book.worksheets.add(name);
  const range = sheet.getRangeByIndexes(0,0,values.length,cols.length);
  range.values = values;
  tables.push({name,range,values});
}
book.recalculate();
const inspect = await book.inspect({kind:'region',sheetId:'i3_summary',range:'A1:P5',maxChars:2500,tableMaxRows:5,tableMaxCols:16});
console.log(inspect.ndjson);
const quote = v => {const s = v == null ? '' : String(v);return /[,"\r\n]/.test(s) ? '"'+s.replaceAll('"','""')+'"' : s;};
const report=[];
for (const {name,range,values} of tables) {
  const actual = range.values;
  if (JSON.stringify(actual) !== JSON.stringify(values)) throw Error('Typed workbook values changed '+name);
  const csv = '\uFEFF'+actual.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n';
  const dest=path.join(out,name+'.csv');
  await writeReviewed(dest,csv);
  report.push({file:name+'.csv',data_rows:actual.length-1,columns:actual[0].length,sha256:crypto.createHash('sha256').update(csv).digest('hex')});
}
await writeReviewed(path.join(out,'csv_export_audit.json'),JSON.stringify({authoring:'@oai/artifact-tool typed values with RFC4180 serialization; no rounding',tables:report},null,2));
console.log(JSON.stringify(report));

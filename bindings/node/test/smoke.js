'use strict';
// Smoke test of the Node binding: version, errors, lists, and (with --record) a short session
// without processing, printing its events.
//
//   node bindings/node/test/smoke.js [--record] [--seconds N] [--config JSON]
const sd = require('..');

let failures = 0;
function check(ok, what) {
  console.log((ok ? 'ok   ' : 'FAIL ') + what);
  if (!ok) failures++;
}

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

async function main() {
  console.log(`sonicdiar ${sd.version()} (node ${process.version})`);
  try {
    sd.listDevices('sideways');
    check(false, 'bad device kind is rejected');
  } catch (e) {
    check(/input/.test(e.message), `bad device kind is rejected (${e.message})`);
  }
  try {
    new sd.Session({ sytem: 'none' });
    check(false, 'unknown config field is rejected');
  } catch (e) {
    check(/sytem/.test(e.message), 'unknown config field is rejected');
  }
  const models = sd.listModels(arg('--models', 'models'));
  check(Array.isArray(models) && models.length > 0, `${models.length} models listed`);
  try {
    console.log('     outputs:', sd.listDevices('output').map((d) => d.name));
  } catch (e) {
    console.log(`     (no device list here: ${e.message})`);
  }

  if (process.argv.includes('--record')) {
    const session = new sd.Session(arg('--config', '{"processing": false}'));
    let started = 0;
    for await (const event of session.events({ seconds: Number(arg('--seconds', '3')) })) {
      console.log('     event:', JSON.stringify(event).slice(0, 160));
      if (event.type === 'device_started') started++;
    }
    check(session.summary !== null, `summary: ${JSON.stringify(session.summary).slice(0, 200)}`);
    check(started > 0, 'a device started');
    try {
      await session.stop();
      check(false, 'stopping twice is an error');
    } catch {
      check(true, 'stopping twice is an error');
    }
  }
  console.log(failures ? 'SMOKE FAILED' : 'SMOKE PASSED');
  process.exit(failures ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

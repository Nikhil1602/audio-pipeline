# SonicDiar (Node.js)

Meeting audio capture for Node.js and Electron: microphone and system audio on Windows, macOS
and Linux, with echo cancellation, voice activity detection, live transcription in 15
languages and speaker labels. Prebuilt for win32-x64, linux-x64, darwin-arm64 and darwin-x64.

```js
const sd = require('sonicdiar');

await sd.fetchModels('models', 'core');                 // once (~50 MB)
await sd.fetchModels('models', 'asr', 'whisper-small');  // for transcription (~490 MB)

const session = new sd.Session({ transcription: { whisper_model: 'ggml-small.bin' } });
for await (const event of session.events({ seconds: 30 })) {
  if (event.type === 'final') console.log(event.source, event.speaker, event.text);
}
console.log(session.summary);
```

Waiting for events and stopping run off the JavaScript thread (Promises). See the repository's
`docs/sdk.md` for the configuration, every event and the platform notes.

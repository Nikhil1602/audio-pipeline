//! Research probe: does `IAudioClient2::SetClientProperties` still apply after
//! `IAudioClient::Initialize`? (Matters for the mic stream category.)
//!
//! Opens the default capture device twice and sets the Communications category, once before
//! and once after `Initialize`, printing the HRESULT of each call. Nothing is captured.
//!
//! `cargo run -p sd-capture --example client_props_order`

#[cfg(windows)]
fn main() -> windows::core::Result<()> {
    use windows::Win32::Media::Audio::{
        AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
        AUDCLNT_STREAMFLAGS_EVENTCALLBACK, AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
        AUDCLNT_STREAMOPTIONS_NONE, AudioCategory_Communications, AudioClientProperties,
        IAudioClient, IAudioClient2, IMMDeviceEnumerator, MMDeviceEnumerator, eCapture, eConsole,
    };
    use windows::Win32::System::Com::{
        CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
    };
    use windows::core::Interface;

    let props = AudioClientProperties {
        cbSize: std::mem::size_of::<AudioClientProperties>() as u32,
        bIsOffload: false.into(),
        eCategory: AudioCategory_Communications,
        Options: AUDCLNT_STREAMOPTIONS_NONE,
    };
    let flags = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
        | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
        | AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
    // SAFETY: plain COM calls on this thread; the mix format is freed after Initialize copies it.
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
        let en: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
        for before in [true, false] {
            let dev = en.GetDefaultAudioEndpoint(eCapture, eConsole)?;
            let client: IAudioClient = dev.Activate(CLSCTX_ALL, None)?;
            let c2 = client.cast::<IAudioClient2>()?;
            let set = || {
                c2.SetClientProperties(&props)
                    .map_or_else(|e| e.code(), |_| Default::default())
            };
            let mut set_hr = Default::default();
            if before {
                set_hr = set();
            }
            let mix = client.GetMixFormat()?;
            let init = client.Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 10_000_000, 0, mix, None);
            CoTaskMemFree(Some(mix as *const _));
            if !before {
                set_hr = set();
            }
            println!(
                "SetClientProperties(Communications) {} Initialize: SetClientProperties -> {set_hr:?}, Initialize -> {:?}",
                if before { "before" } else { "after " },
                init.map_or_else(|e| e.code(), |_| Default::default()),
            );
        }
    }
    Ok(())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("Windows only");
}

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MultiDocChat</title>
<link rel="stylesheet" href="/static/styles.css" />
<meta name="color-scheme" content="light dark" />

<script>
const $ = (sel) => document.querySelector(sel);
let sessionId = localStorage.getItem('mdc_session_id') || '';

async function uploadFiles(evt) {
    evt.preventDefault();
    const input = $('#file-input');
    const files = input.files;
    if (!files || files.length === 0) {
        toast('Please choose at least one file');
        return;
    }
    toggleIndexing(true);
    try {
        const fd = new FormData();
        for (const f of files) fd.append('files', f);
        const key = localStorage.getItem('OPENROUTER_API_KEY');
        fd.append('openrouter_key', key); // send key to backend

        const res = await fetch('/upload', { method: 'POST', body: fd });
        if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed');
        const data = await res.json();
        sessionId = data.session_id;
        localStorage.setItem('mdc_session_id', sessionId);
        $('#chat').style.display = 'block';
        toast('Indexing complete. You can chat now.');
    } catch (e) {
        console.error(e);
        toast('Indexing failed. Try re-uploading.');
    } finally {
        toggleIndexing(false);
    }
}

async function sendMessage() {
    const input = $('#message-input');
    const text = input.value.trim();
    if (!sessionId) { toast('Please upload documents first.'); return; }
    if (!text) return;
    appendMessage('user', text);
    input.value = '';
    toggleThinking(true);
    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                session_id: sessionId,
                message: text,
                openrouter_key: localStorage.getItem('OPENROUTER_API_KEY')
            })
        });
        if (!res.ok) throw new Error((await res.json()).detail || 'Chat failed');
        const data = await res.json();
        appendMessage('assistant', data.answer);
    } catch (e) {
        console.error(e);
        toast('Thinking failed. Try again.');
    } finally {
        toggleThinking(false);
    }
}

function appendMessage(role, text) {
    const container = $('#messages');
    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + (role === 'user' ? 'user' : 'assistant');
    bubble.textContent = text;
    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
}

function toggleIndexing(on) { $('#upload-btn').disabled = on; $('#indexing').style.display = on ? 'inline-block' : 'none'; }
function toggleThinking(on) { $('#send-btn').disabled = on; $('#thinking').style.display = on ? 'inline-block' : 'none'; }

function toast(msg) {
    const el = $('#toast');
    el.textContent = msg;
    el.style.opacity = '1';
    setTimeout(() => el.style.opacity = '0', 2500);
}

window.addEventListener('DOMContentLoaded', () => {
    if (sessionId) $('#chat').style.display = 'block';

    $('#upload-form').addEventListener('submit', uploadFiles);
    $('#send-btn').addEventListener('click', sendMessage);
    $('#save-key-btn').addEventListener('click', () => {
        const key = $('#openrouter-key').value.trim();
        if (!key) { toast('Please enter a valid OpenRouter key.'); return; }
        localStorage.setItem('OPENROUTER_API_KEY', key);
        toast('OpenRouter key saved!');
    });

    $('#message-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    const drop = $('#dropzone');
    drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('hover'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('hover'));
    drop.addEventListener('drop', (e) => {
        e.preventDefault(); drop.classList.remove('hover');
        $('#file-input').files = e.dataTransfer.files;
        const preview = $('#file-preview'); preview.innerHTML = '';
        for (const f of e.dataTransfer.files) {
            const div = document.createElement('div'); div.className='file-item'; div.textContent=f.name; preview.appendChild(div);
        }
        preview.style.display='block';
    });

    $('#file-input').addEventListener('change', function() {
        const preview = $('#file-preview'); preview.innerHTML = '';
        for (const file of this.files) {
            const div = document.createElement('div'); div.className='file-item'; div.textContent=file.name; preview.appendChild(div);
        }
        preview.style.display='block';
    });
});
</script>
</head>

<body>
<header>
    <h1>MultiDocChat</h1>
</header>

<main>
    <!-- OPENROUTER KEY INPUT -->
    <section id="key-section">
        <input type="password" id="openrouter-key" placeholder="Enter OpenRouter API Key" />
        <button id="save-key-btn" class="btn">Save Key</button>
    </section>

    <!-- FILE UPLOADER -->
    <section id="uploader">
        <form id="upload-form">
            <div id="dropzone">
                <p>Drag & Drop Files</p>
                <p>or</p>
                <label class="btn">
                    Choose Files
                    <input id="file-input" type="file" name="files" multiple hidden />
                </label>
            </div>
            <div id="file-preview"></div>
            <button id="upload-btn" type="submit">Upload & Index</button>
            <span id="indexing" class="hint" style="display:none;">Indexing…</span>
        </form>
    </section>

    <!-- CHAT -->
    <section id="chat" style="display:none;">
        <div id="messages" class="messages"></div>
        <div class="composer">
            <textarea id="message-input" rows="2" placeholder="Ask about your documents…"></textarea>
            <button id="send-btn">Send</button>
            <span id="thinking" class="hint" style="display:none;">Thinking…</span>
        </div>
    </section>
</main>

<div id="toast" class="toast"></div>
</body>
</html>

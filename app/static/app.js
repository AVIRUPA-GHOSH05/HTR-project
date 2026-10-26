(() => {
  const picker = document.querySelector('#file-input');
  const dropzone = document.querySelector('#dropzone');
  const previewWrap = document.querySelector('#preview-wrap');
  const preview = document.querySelector('#preview');
  const fileName = document.querySelector('#file-name');
  const recognize = document.querySelector('#recognize');
  const buttonLabel = document.querySelector('#button-label');
  const spinner = document.querySelector('#spinner');
  const output = document.querySelector('#result');
  const message = document.querySelector('#message');
  const maxBytes = 10 * 1024 * 1024;
  let selected = null;

  function say(text, tone = '') { message.textContent = text; message.className = `message ${tone}`.trim(); }
  function choose(file) {
    if (!file) return;
    const accepted = ['image/png', 'image/jpeg', 'image/bmp'];
    if (!accepted.includes(file.type) || file.size > maxBytes) {
      selected = null;
      recognize.disabled = true;
      previewWrap.hidden = true;
      dropzone.hidden = false;
      preview.removeAttribute('src');
      say(file.size > maxBytes ? 'That image is over the 10 MB limit.' : 'Choose a PNG, JPG, or BMP image.', 'error');
      return;
    }
    selected = file;
    preview.src = URL.createObjectURL(file);
    fileName.textContent = file.name;
    dropzone.hidden = true;
    previewWrap.hidden = false;
    recognize.disabled = false;
    say('Image ready. Select recognize when you’re ready.');
  }
  dropzone.addEventListener('click', () => picker.click());
  picker.addEventListener('change', () => choose(picker.files[0]));
  for (const eventName of ['dragenter', 'dragover']) dropzone.addEventListener(eventName, event => { event.preventDefault(); dropzone.classList.add('dragging'); });
  for (const eventName of ['dragleave', 'drop']) dropzone.addEventListener(eventName, event => { event.preventDefault(); dropzone.classList.remove('dragging'); });
  dropzone.addEventListener('drop', event => choose(event.dataTransfer.files[0]));
  document.querySelector('#remove-image').addEventListener('click', () => {
    selected = null; picker.value = ''; preview.removeAttribute('src'); previewWrap.hidden = true; dropzone.hidden = false; recognize.disabled = true; say('Image removed.');
  });
  recognize.addEventListener('click', async () => {
    if (!selected) return;
    recognize.disabled = true; spinner.hidden = false; buttonLabel.textContent = 'Recognizing…'; say('Reading the handwriting…');
    const form = new FormData(); form.append('file', selected);
    try {
      const response = await fetch('/predict', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Recognition failed.');
      output.value = data.text;
      say('Recognition complete. You can edit the text below.', 'success');
    } catch (error) { say(error.message || 'Could not reach the server. Please try again.', 'error'); }
    finally { recognize.disabled = !selected; spinner.hidden = true; buttonLabel.textContent = 'Recognize handwriting'; }
  });
  document.querySelector('#copy').addEventListener('click', async () => {
    if (!output.value) return say('There is no text to copy yet.', 'error');
    try { await navigator.clipboard.writeText(output.value); say('Text copied to clipboard.', 'success'); }
    catch { output.select(); document.execCommand('copy'); say('Text copied to clipboard.', 'success'); }
  });
  document.querySelector('#clear').addEventListener('click', () => { output.value = ''; say('Transcription cleared.'); });
  document.querySelector('#pdf').addEventListener('click', async () => {
    if (!output.value.trim()) return say('Add or recognize some text before downloading a PDF.', 'error');
    try {
      const response = await fetch('/download-pdf', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: output.value }) });
      if (!response.ok) throw new Error('The PDF could not be generated.');
      const blob = await response.blob(); const url = URL.createObjectURL(blob); const link = document.createElement('a');
      link.href = url; link.download = 'transcription.pdf'; link.click(); URL.revokeObjectURL(url); say('Your PDF is ready.', 'success');
    } catch (error) { say(error.message, 'error'); }
  });
})();

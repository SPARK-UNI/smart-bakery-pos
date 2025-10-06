// web/script.js
// Đổi tên hiển thị và thông báo; giữ nguyên toàn bộ logic camera + fetch /predict
class BakeryRecognition {
  constructor() {
    // --- phần có sẵn ---
    this.video = document.getElementById('video');
    this.startBtn = document.getElementById('startBtn');
    this.predictBtn = document.getElementById('predictBtn');
    this.stopBtn = document.getElementById('stopBtn');
    this.result = document.getElementById('result');
    this.status = document.getElementById('status');
    this.statusDot = document.querySelector('.status-dot');
    this.loading = document.getElementById('loading');
    this.placeholder = document.getElementById('placeholder');
    this.stream = null;
    this.isProcessing = false;

    // --- thêm mới: phần tử upload ---
    this.uploadBtn = document.getElementById('uploadBtn');
    this.fileInput = document.getElementById('fileInput');
    this.previewWrap = document.getElementById('uploadPreviewWrap');
    this.previewImg = document.getElementById('uploadPreview');

    this.init();
  }

  init() {
    // camera
    this.startBtn.addEventListener('click', () => this.startCamera());
    this.stopBtn.addEventListener('click', () => this.stopCamera());
    this.predictBtn.addEventListener('click', () => this.predict());

    // upload
    this.uploadBtn.addEventListener('click', () => this.fileInput.click());
    this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files));

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.updateResult('Trình duyệt không hỗ trợ camera', 'error');
      this.startBtn.disabled = true;
    }
  }

  async startCamera() { /* giữ nguyên như bản cũ */ 
    try {
      this.startBtn.disabled = true;
      this.updateResult('Đang khởi động camera...', '');
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
      });
      this.video.srcObject = this.stream;
      this.placeholder.classList.add('hidden');
      this.video.onloadedmetadata = () => {
        this.updateStatus(true);
        this.predictBtn.disabled = false;
        this.stopBtn.disabled = false;
        this.updateResult('Camera đã sẵn sàng', 'success');
      };
    } catch (e) { this.handleCameraError(e); }
  }

  stopCamera() { /* giữ nguyên như bản cũ */ 
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    this.stream = null; this.video.srcObject = null;
    this.placeholder.classList.remove('hidden');
    this.startBtn.disabled = false; this.predictBtn.disabled = true; this.stopBtn.disabled = true;
    this.updateStatus(false); this.updateResult('Camera đã tắt', '');
  }

  async predict() {
    if (this.isProcessing || !this.stream) return;
    this.isProcessing = true; this.predictBtn.disabled = true; this.showLoading(true);
    try {
      const imageData = this.captureFrame();
      const res = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      });
      const data = await res.json();
      this.updateResult(data.prediction || 'Không có kết quả', data.error ? 'error' : 'success');
    } catch (e) {
      this.updateResult('Không thể kết nối server', 'error');
    } finally {
      this.isProcessing = false; this.predictBtn.disabled = false; this.showLoading(false);
    }
  }

  // --- thêm mới: chọn file, hiển thị preview và gọi predict ---
  handleFileSelect(files) {
    if (!files || !files.length) return;
    const file = files[0];
    if (!file.type.startsWith('image/')) {
      this.updateResult('File không phải ảnh', 'error');
      return;
    }

    const reader = new FileReader();
    reader.onload = async () => {
      const dataURL = reader.result;  // base64
      // show preview
      this.previewImg.src = dataURL;
      this.previewWrap.classList.remove('hidden');

      // gửi tới /predict, tái dùng loading/result cũ
      this.showLoading(true);
      try {
        const res = await fetch('http://127.0.0.1:5000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataURL })
        });
        const data = await res.json();
        this.updateResult(data.prediction || 'Không có kết quả', data.error ? 'error' : 'success');
      } catch (err) {
        this.updateResult('Không thể kết nối server', 'error');
      } finally {
        this.showLoading(false);
        this.fileInput.value = ''; // cho phép chọn lại cùng file
      }
    };
    reader.readAsDataURL(file);
  }

  captureFrame() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = this.video.videoWidth; canvas.height = this.video.videoHeight;
    ctx.drawImage(this.video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8);
  }

  updateStatus(on) {
    this.status.innerHTML = `<span class="status-dot ${on ? 'online':'offline'}"></span>${on ? 'Camera Online' : 'Camera Offline'}`;
    this.statusDot.classList.toggle('online', on);
    this.statusDot.classList.toggle('offline', !on);
  }

  updateResult(text, type) { this.result.textContent = text; this.result.className = `result-text ${type}`; }
  showLoading(show) { this.loading.classList.toggle('hidden', !show); this.result.style.display = show ? 'none':'block'; }

  handleCameraError(e) {
    let msg = 'Không thể truy cập camera';
    if (e.name === 'NotAllowedError') msg = 'Vui lòng cho phép truy cập camera';
    else if (e.name === 'NotFoundError') msg = 'Không tìm thấy camera';
    else if (e.name === 'NotSupportedError') msg = 'Trình duyệt không hỗ trợ camera';
    this.updateResult(msg, 'error'); this.startBtn.disabled = false;
  }
}
document.addEventListener('DOMContentLoaded', () => { window.app = new BakeryRecognition(); });
window.addEventListener('beforeunload', () => { if (window.app?.stream) window.app.stopCamera(); });

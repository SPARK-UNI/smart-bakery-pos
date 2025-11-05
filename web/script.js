// === script.js (Gộp nút nhưng giữ nguyên toàn bộ chức năng) ===
class BakeryRecognition {
  constructor() {
    this.apiURL = (window.location.origin || 'http://127.0.0.1:5000');

    // Elements
    this.video = document.getElementById('video');
    this.viewportImg = document.getElementById('viewportImg');

    // Nút gộp
    this.cameraToggleBtn = document.getElementById('cameraToggleBtn');
    this.captureToggleBtn = document.getElementById('captureToggleBtn');

    // Upload
    this.uploadBtn = document.getElementById('uploadBtn');
    this.fileInput = document.getElementById('fileInput');

    this.placeholder = document.getElementById('placeholder');
    this.status = document.getElementById('status');

    this.trayGrid = document.getElementById('trayGrid');
    this.billItems = document.getElementById('billItems');
    this.billTotal = document.getElementById('billTotal');

    // payment
    this.payCash = document.getElementById('payCash');
    this.payTransfer = document.getElementById('payTransfer');
    this.cashGiven = document.getElementById('cashGiven');
    this.changeDue = document.getElementById('changeDue');
    this.payBtn = document.getElementById('payBtn');
    this.payStatus = document.getElementById('payStatus');

    // State
    this.stream = null;
    this.traySlices = [];
    this.trayOverrides = [];
    this.mode = 'camera'; // 'camera' | 'image'

    // Prices (display keys are the dropdown values)
    this.PRICE_MAP = {
      'Egg Tart': 21000,
      'Croissant': 30000,
      'Cookies dừa': 23000,
      'Chà bông cây': 27000,
      'Patechaud': 30000,
      'Bánh mì dừa lưới': 15000,
      'Bánh da lợn': 23000,
      'Bánh mì bơ ( Cua lớn )': 18000,
      'Muffin Việt Quất': 25000,
      'Bánh chuối nướng': 19000
    };
    this.LABEL_OPTIONS = Object.keys(this.PRICE_MAP);

    // Build canonical map to match any model label (lowercase, no accents) to dropdown value
    this.CANONICAL = this.buildCanonical();
    this.CONF_THRESHOLD = 0.50;

    this.init();
  }

  // ===== Canonicalization helpers =====
  buildCanonical(){
    const map = new Map();
    const add = (k, disp) => { if (!map.has(k)) map.set(k, disp); };

    this.LABEL_OPTIONS.forEach(disp => {
      const base = this.normalizeVN(disp);
      add(base, disp);
      add(base.replace(/[()]/g, '').replace(/\s+/g, ' ').trim(), disp);
    });
    // Common aliases (lowercase without accents)
    add('croissant', 'Croissant');
    add('egg tart','Egg Tart');
    add('patechaud','Patechaud');
    add('cookies dua','Cookies dừa');
    add('cha bong cay','Chà bông cây');
    add('banh mi dua luoi','Bánh mì dừa lưới');
    add('banh da lon','Bánh da lợn');
    add('muffin viet quat','Muffin Việt Quất');
    add('banh chuoi nuong','Bánh chuối nướng');
    add('banh cua bo','Bánh mì bơ ( Cua lớn )');
    return map;
  }
  normalizeVN(str){
    if (!str) return '';
    const s = str.normalize('NFD').replace(/[\u0300-\u036f]/g,'')
      .replace(/đ/g,'d').replace(/Đ/g,'D').toLowerCase();
    return s.replace(/[^a-z0-9\s()]/g,' ').replace(/\s+/g,' ').trim();
  }
  canonicalFromPred(predLabel){
    const n = this.normalizeVN(predLabel);
    if (this.CANONICAL.has(n)) return this.CANONICAL.get(n);
    const n2 = n.replace(/[()]/g,'').replace(/\s+/g,' ').trim();
    if (this.CANONICAL.has(n2)) return this.CANONICAL.get(n2);
    return null;
  }

  init(){
    // Toggle camera (Bật/Tắt)
    this.cameraToggleBtn?.addEventListener('click', ()=>this.toggleCamera());
    // Toggle capture (Chụp khay/Trở về camera)
    this.captureToggleBtn?.addEventListener('click', ()=>this.toggleCapture());

    // Upload
    this.uploadBtn?.addEventListener('click', ()=>this.fileInput.click());
    this.fileInput?.addEventListener('change', e=>this.handleUpload(e.target.files));

    // Payment
    document.querySelectorAll('input[name="pay"]').forEach(r=>r.addEventListener('change', ()=>this.onPayMethodChange()));
    this.cashGiven?.addEventListener('input', ()=>this.updateChangeDue());
    this.payBtn?.addEventListener('click', ()=>this.onPay());

    // Khởi tạo trạng thái nút
    this.setCaptureBtnEnabled(false);
  }

  // ======== Toggle buttons ========
  async toggleCamera(){
    if (!this.stream){
      await this.startCamera();
    } else {
      this.stopCamera();
    }
  }
  async toggleCapture(){
    if (this.mode === 'camera'){
      await this.handleCapture();       // chụp + nhận diện
      this.captureToggleBtn.textContent = 'Trở về camera';
    } else {
      this.showCamera();                // quay lại video
      this.captureToggleBtn.textContent = 'Chụp khay';
    }
  }

  // ======== View state helpers ========
  showImage(dataURL){
    this.viewportImg.src = dataURL;
    this.viewportImg.classList.remove('hidden');
    this.video.classList.add('hidden');
    this.mode = 'image';
  }
  showCamera(){
    this.viewportImg.classList.add('hidden');
    this.video.classList.remove('hidden');
    this.mode = 'camera';
  }

  // ======== Camera control ========
  async startCamera(){
    try{
      this.cameraToggleBtn.disabled = true;
      this.cameraToggleBtn.textContent = 'Đang bật...';
      this.stream = await navigator.mediaDevices.getUserMedia({ video:{ width:{ideal:1280}, height:{ideal:720}, facingMode:'environment' } });
      this.video.srcObject = this.stream;
      this.placeholder.classList.add('hidden');
      this.updateStatus(true);
      this.setCaptureBtnEnabled(true);
      this.cameraToggleBtn.textContent = 'Tắt Camera';
      if (this.mode !== 'image') this.showCamera();
    }catch(e){
      alert('Không thể mở camera');
    } finally {
      this.cameraToggleBtn.disabled = false;
    }
  }
  stopCamera(){
    if (this.stream) this.stream.getTracks().forEach(t=>t.stop());
    this.stream=null; this.video.srcObject=null;
    this.placeholder.classList.remove('hidden');
    this.updateStatus(false);
    // Khi tắt camera, nếu đang ở chế độ image thì giữ nguyên ảnh;
    // nhưng nút chụp không còn tác dụng -> disable
    this.setCaptureBtnEnabled(false);
    this.cameraToggleBtn.textContent = 'Bật Camera';
  }
  setCaptureBtnEnabled(on){
    this.captureToggleBtn.disabled = !on;
    if (!on) this.captureToggleBtn.textContent = 'Chụp khay';
  }
  updateStatus(on){
    this.status.innerHTML = `<span class="status-dot ${on?'online':'offline'}"></span>${on?'Camera Online':'Camera Offline'}`;
  }

  // ======== Chụp & Nhận diện ========
  async handleCapture(){
    // Chỉ chụp khi đang có camera
    if (!this.stream){ return; }
    const dataURL = this.captureFrame();
    this.showImage(dataURL);
    const layout = document.getElementById('trayLayout')?.value || 'auto6';
    await this.splitRecognizeAndRender(dataURL, layout);
  }
  captureFrame(){
    const canvas=document.createElement('canvas');
    const ctx=canvas.getContext('2d');
    const w=this.video.videoWidth||640, h=this.video.videoHeight||480;
    canvas.width=w; canvas.height=h; ctx.drawImage(this.video,0,0,w,h);
    return canvas.toDataURL('image/jpeg',0.95);
  }
  handleUpload(files){
    if (!files||!files.length) return;
    const file=files[0];
    if (!file.type.startsWith('image/')) { alert('File không phải ảnh'); return; }
    const reader=new FileReader();
    reader.onload=async ()=>{
      const dataURL=reader.result;
      this.showImage(dataURL);
      const layout = document.getElementById('trayLayout')?.value || 'auto6';
      await this.splitRecognizeAndRender(dataURL, layout);
      // Cho phép quay về camera nếu camera đang bật
      if (this.stream){ this.captureToggleBtn.textContent = 'Trở về camera'; this.captureToggleBtn.disabled = false; }
    };
    reader.readAsDataURL(file);
  }

  async splitRecognizeAndRender(dataURL, layout='auto6'){
    const slices = await this.splitTray(dataURL, layout);
    this.traySlices = slices;
    this.renderTrayGrid(true);

    const res = await this.postJSON('/predict_batch', { images: slices, center_crop: false });
    if (!res || !res.results) return;

    res.results.forEach((r,i)=>{
      // UI prediction (label + conf + suggestions)
      this.renderPrediction(i, r);

      // Auto-select when confident: map to dropdown display name first
      const canonical = this.canonicalFromPred(r.label || '');
      if (typeof r.confidence === 'number' && r.confidence >= this.CONF_THRESHOLD && canonical && this.PRICE_MAP.hasOwnProperty(canonical)) {
        const sel = document.getElementById(`override-${i}`);
        sel.value = canonical;                            // ✅ update dropdown visible value
        this.trayOverrides[i] = canonical;                // ✅ update internal
        sel.dispatchEvent(new Event('change'));           // ✅ trigger bill update
      } else {
        this.trayOverrides[i] = null;
      }
    });
  }

  async splitTray(dataURL, layout='auto6'){
    return new Promise((resolve)=>{
      const img=new Image();
      img.onload=()=>{
        const W=img.naturalWidth, H=img.naturalHeight;
        const slices=[]; let boxes=[];

        if (layout==='auto6'){
          const rows=2, cols=3;
          const tileW=Math.floor(W/cols), tileH=Math.floor(H/rows);
          for (let r=0;r<rows;r++){
            for (let c=0;c<cols;c++){
              boxes.push({x:c*tileW, y:r*tileH, w:tileW, h:tileH});
            }
          }
        }else if (layout==='manual5'){
          boxes=[
            { x: 0.07*W, y: 0.06*H, w: 0.27*W, h: 0.35*H },
            { x: 0.355*W, y: 0.06*H, w: 0.27*W, h: 0.35*H },
            { x: 0.63*W, y: 0.06*H, w: 0.27*W, h: 0.35*H },
            { x: 0.059*W, y: 0.417*H, w: 0.341*W, h: 0.525*H },
            { x: 0.498*W, y: 0.417*H, w: 0.427*W, h: 0.525*H }
          ];
        }

        boxes.forEach((b)=>{
          const c=document.createElement('canvas');
          c.width=b.w; c.height=b.h;
          const cx=c.getContext('2d');
          cx.drawImage(img, b.x,b.y,b.w,b.h, 0,0,b.w,b.h);
          slices.push(c.toDataURL('image/jpeg',0.95));
        });
        resolve(slices);
      };
      img.src=dataURL;
    });
  }

  renderPrediction(i, r){
    const predEl=document.getElementById(`pred-${i}`);
    predEl.innerHTML='';

    // Always show label + conf
    const labTxt = (r.label || '').toLowerCase();
    const confTxt = (typeof r.confidence==='number') ? r.confidence.toFixed(2) : '';
    const header = document.createElement('div');
    header.className = 'pred-header';
    header.innerHTML = `<span>${labTxt}</span> <span class="conf">(${confTxt})</span>`;
    predEl.appendChild(header);

    // Suggestions (clickable) when not confident
    if (typeof r.confidence==='number' && r.confidence < this.CONF_THRESHOLD && Array.isArray(r.topk) && r.topk.length){
      const sugWrap = document.createElement('div');
      sugWrap.className = 'pred-sugs';
      r.topk.forEach(([rawName, score])=>{
        const dispName = this.canonicalFromPred(rawName) || rawName;
        const btn = document.createElement('button');
        btn.className = 'sug-btn';
        btn.textContent = `${dispName} (${Number(score).toFixed(2)})`;
        btn.addEventListener('click', ()=>{
          const sel=document.getElementById(`override-${i}`);
          sel.value = dispName;                 // set dropdown
          this.trayOverrides[i] = dispName;     // update internal
          header.innerHTML = `<span>${dispName.toLowerCase()}</span> <span class="conf">(chọn: ${Number(score).toFixed(2)})</span>`;
          sel.dispatchEvent(new Event('change')); // bill update
        });
        sugWrap.appendChild(btn);
      });
      predEl.appendChild(sugWrap);
    }
  }

  renderTrayGrid(loading=false){
    this.trayGrid.innerHTML='';
    this.trayOverrides = new Array(this.traySlices.length).fill(null);
    for (let i=0;i<this.traySlices.length;i++){
      const tile=document.createElement('div'); tile.className='tray-tile';
      const img=document.createElement('img'); img.className='tray-img'; img.src=this.traySlices[i]||'';
      const pred=document.createElement('div'); pred.className='tray-pred'; pred.id=`pred-${i}`; pred.innerHTML=loading?'Đang nhận diện...':'Chưa nhận diện';
      const select=document.createElement('select'); select.className='tray-select'; select.id=`override-${i}`;
      const empty=document.createElement('option'); empty.value=''; empty.textContent='— chỉnh loại bánh —'; select.appendChild(empty);
      this.LABEL_OPTIONS.forEach(name=>{ const op=document.createElement('option'); op.value=name; op.textContent=name; select.appendChild(op); });
      select.addEventListener('change', ()=>{ this.trayOverrides[i]=select.value||null; this.updateBill(); });
      tile.appendChild(img); tile.appendChild(pred); tile.appendChild(select);
      this.trayGrid.appendChild(tile);
    }
    this.updateBill();
  }

  updateBill(){
    const counts={};
    for (let label of this.trayOverrides){
      if (!label) continue;
      counts[label]=(counts[label]||0)+1;
    }
    this.billItems.innerHTML='';
    let total=0;
    Object.entries(counts).forEach(([name,qty])=>{
      const price=this.PRICE_MAP[name]||0;
      const row=document.createElement('div'); row.className='row';
      row.innerHTML=`<span>${name}</span><span class="qty">x${qty}</span><span class="price">${(price*qty).toLocaleString('vi-VN')}</span>`;
      this.billItems.appendChild(row);
      total+=price*qty;
    });
    this.billTotal.textContent = total.toLocaleString('vi-VN');
    this.updateChangeDue();
  }

  onPayMethodChange(){
    const method = (document.querySelector('input[name="pay"]:checked')?.value) || 'cash';
    this.payCash?.classList.toggle('hidden', method!=='cash');
    this.payTransfer?.classList.toggle('hidden', method!=='transfer');
    this.payStatus.textContent='';
  }
  updateChangeDue(){
    const method = (document.querySelector('input[name="pay"]:checked')?.value) || 'cash';
    const total = this.parseVND(this.billTotal.textContent);
    if (method==='cash'){
      const given = parseInt(this.cashGiven?.value||'0',10);
      const change = Math.max(given - total, 0);
      if (this.changeDue) this.changeDue.textContent = change.toLocaleString('vi-VN');
    } else {
      if (this.changeDue) this.changeDue.textContent = '0';
    }
  }
  parseVND(str){ return parseInt(String(str).replace(/[^0-9]/g,''),10) || 0; }
  onPay(){
    const method = (document.querySelector('input[name="pay"]:checked')?.value) || 'cash';
    const total = this.parseVND(this.billTotal.textContent);
    if (total<=0){ this.payStatus.textContent='Chưa có mặt hàng trong hóa đơn.'; return; }
    if (method==='cash'){
      const given = this.parseVND(this.cashGiven?.value||'0');
      if (given < total){ this.payStatus.textContent='Số tiền khách đưa chưa đủ.'; return; }
      this.payStatus.textContent = 'Đã thanh toán tiền mặt ✔';
    } else if (method==='card'){
      this.payStatus.textContent = 'Đã thanh toán qua thẻ ✔';
    } else {
      this.payStatus.textContent = 'Đang chờ khách chuyển khoản... ✔';
    }
  }

  async postJSON(path, body){
    const res = await fetch(this.apiURL + path, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body||{}) });
    return await res.json();
  }
}

document.addEventListener('DOMContentLoaded', ()=>{ window.app = new BakeryRecognition(); });


let tempChart, humChart, batChart;
let labels = [];
let tempData = [], humData = [], batData = [];
let deviceStatus = {};

function createCharts(){
  const ctxT = document.getElementById('chartTemp').getContext('2d');
  tempChart = new Chart(ctxT, {
    type: 'line',
    data: { labels: labels, datasets: [{ label: 'Temperature', data: tempData, fill:false, borderWidth:2, pointRadius:0, borderColor:'#00f0ff' }]},
    options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{display:false}}, scales:{ x:{ display:false } } }
  });
  const ctxH = document.getElementById('chartHum').getContext('2d');
  humChart = new Chart(ctxH, {
    type: 'line',
    data: { labels: labels, datasets: [{ label: 'Humidity', data: humData, fill:false, borderWidth:2, pointRadius:0, borderColor:'#9b59ff' }]},
    options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{display:false}}, scales:{ x:{ display:false } } }
  });
  const ctxB = document.getElementById('chartBat').getContext('2d');
  batChart = new Chart(ctxB, {
    type: 'line',
    data: { labels: labels, datasets: [{ label: 'Battery', data: batData, fill:false, borderWidth:2, pointRadius:0, borderColor:'#39ff7b' }]},
    options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{display:false}}, scales:{ x:{ display:false } } }
  });
}

function updateCharts(t, h, b){
  const timeLabel = new Date().toLocaleTimeString();
  labels.push(timeLabel);
  if(labels.length>30){ labels.shift(); tempData.shift(); humData.shift(); batData.shift(); }
  tempData.push(t); humData.push(h); batData.push(b);
  tempChart.update(); humChart.update(); batChart.update();
}

function updateDeviceTable(device, t, h, b, status){
  deviceStatus[device] = {t,h,b,status};
  const tbody = document.querySelector('#deviceTable tbody');
  tbody.innerHTML = '';
  Object.keys(deviceStatus).forEach(d => {
    const r = deviceStatus[d];
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${d}</td><td>${r.t.toFixed(2)}</td><td>${r.h.toFixed(2)}</td><td>${r.b.toFixed(2)}</td><td>${ r.status ? '<span class="status-dot dot-red"></span>ANOMALY' : '<span class="status-dot dot-green"></span>NORMAL'}</td>`;
    tbody.appendChild(tr);
  });
}

function pushAlert(msg){
  const alerts = document.getElementById('alerts');
  const div = document.createElement('div');
  div.className='alert';
  div.innerText = `[${new Date().toLocaleTimeString()}] ` + msg;
  alerts.prepend(div);
  while(alerts.children.length>8) alerts.removeChild(alerts.lastChild);
}

async function fetchLatest(){
  try{
    const res = await fetch('/data/latest');
    const j = await res.json();
    if(!j.success){ console.error('No data'); return; }
    const row = j.row;
    const keys = Object.keys(row);
    const tKey = keys.find(k=>/temp/i.test(k)) || keys.find(k=>/latency|temp/i.test(k));
    const hKey = keys.find(k=>/hum/i.test(k)) || keys.find(k=>/jitter|hum/i.test(k));
    const bKey = keys.find(k=>/battery/i.test(k)) || keys.find(k=>/throughput/i.test(k));
    const deviceKey = keys.find(k=>/device/i.test(k)) || keys[0];
    const t = parseFloat(row[tKey]) || 0;
    const h = parseFloat(row[hKey]) || 0;
    const b = parseFloat(row[bKey]) || 0;
    const device = row[deviceKey] || ('dev_'+j.index);
    updateCharts(t,h,b);
    updateDeviceTable(device,t,h,b,j.anomaly);
    if(j.anomaly) { pushAlert(`${device} flagged as anomaly (score=${j.score?j.score.toFixed(3):'n/a'})`); document.getElementById('statusBadge').className='badge red'; document.getElementById('statusBadge').innerText='ALERT'; }
    else { document.getElementById('statusBadge').className='badge green'; document.getElementById('statusBadge').innerText='LIVE'; }
  }catch(e){
    console.error(e);
  }
}

window.addEventListener('load', ()=>{
  createCharts();
  for(let i=0;i<6;i++) setTimeout(fetchLatest, i*200);
  setInterval(fetchLatest, 1200);
});

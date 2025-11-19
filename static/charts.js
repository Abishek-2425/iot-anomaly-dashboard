// static/charts.js - Clean Chart.js logic for real-time simulation (Top 5 devices)

let tempChart, humChart, batChart;
let labels = [];
let tempData = [], humData = [], batData = [];
let deviceStatus = {};

function createCharts(){
  const ctxT = document.getElementById('chartTemp').getContext('2d');
  tempChart = new Chart(ctxT, {
    type: 'line',
    data: { labels: labels, datasets: [{ label: 'Temperature', data: tempData, fill:true, borderWidth:2, pointRadius:0, backgroundColor:'rgba(31,122,219,0.08)', borderColor:'#1f7adb' }]},
    options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{display:false}}, scales:{ x:{ display:false }, y:{ beginAtZero:false } } }
  });

  const ctxH = document.getElementById('chartHum').getContext('2d');
  humChart = new Chart(ctxH, {
    type: 'line',
    data: { labels: labels, datasets: [{ label: 'Humidity', data: humData, fill:true, borderWidth:2, pointRadius:0, backgroundColor:'rgba(155,155,255,0.06)', borderColor:'#6b72ff' }]},
    options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{display:false}}, scales:{ x:{ display:false } } }
  });

  const ctxB = document.getElementById('chartBat').getContext('2d');
  batChart = new Chart(ctxB, {
    type: 'line',
    data: { labels: labels, datasets: [{ label: 'Battery', data: batData, fill:true, borderWidth:2, pointRadius:0, backgroundColor:'rgba(22,163,74,0.06)', borderColor:'#16a34a' }]},
    options: { responsive:true, maintainAspectRatio:false, plugins:{ legend:{display:false}}, scales:{ x:{ display:false }, y:{ beginAtZero:true, max:110 } } }
  });
}

function updateCharts(t, h, b){
  const timeLabel = new Date().toLocaleTimeString();
  labels.push(timeLabel);
  if(labels.length>40){
    labels.shift(); tempData.shift(); humData.shift(); batData.shift();
  }
  tempData.push(t); humData.push(h); batData.push(b);
  tempChart.update(); humChart.update(); batChart.update();
}

function updateDeviceTable(rows){
  const tbody = document.querySelector('#deviceTable tbody');
  tbody.innerHTML = ''; // clear once

  rows.forEach(row => {
    const device = row.Device_ID;
    const keys = Object.keys(row);
    const tKey = keys.find(k=>/temp/i.test(k)) || keys.find(k=>/temperature/i.test(k));
    const hKey = keys.find(k=>/hum/i.test(k)) || keys.find(k=>/humidity/i.test(k));
    const bKey = keys.find(k=>/bat/i.test(k)) || keys.find(k=>/battery/i.test(k));
    const t = tKey ? parseFloat(row[tKey]) : 0;
    const h = hKey ? parseFloat(row[hKey]) : 0;
    const b = bKey ? parseFloat(row[bKey]) : 0;
    const isAnom = row.anomaly ? 1 : 0;

    deviceStatus[device] = {t,h,b,status:isAnom};

    const statusHtml = isAnom ? 
      '<span class="status-dot dot-red"></span><strong style="color:#ef4444">ANOMALY</strong>' :
      '<span class="status-dot dot-green"></span>Normal';

    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${device}</td><td>${t.toFixed(2)}</td><td>${h.toFixed(2)}</td><td>${b.toFixed(2)}</td><td>${statusHtml}</td>`;
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
    if(!j.success){ console.error('No data', j); return; }
    const rows = j.rows;

    // Update charts & device table
    rows.forEach(row => {
      const keys = Object.keys(row);
      const tKey = keys.find(k=>/temp/i.test(k)) || keys.find(k=>/temperature/i.test(k));
      const hKey = keys.find(k=>/hum/i.test(k)) || keys.find(k=>/humidity/i.test(k));
      const bKey = keys.find(k=>/bat/i.test(k)) || keys.find(k=>/battery/i.test(k));
      updateCharts(tKey ? parseFloat(row[tKey]) : 0,
                   hKey ? parseFloat(row[hKey]) : 0,
                   bKey ? parseFloat(row[bKey]) : 0);
    });

    updateDeviceTable(rows);  // pass all rows once

    // Alerts
    rows.forEach(row => {
      if(row.anomaly){
        pushAlert(`${row.Device_ID} flagged as anomaly (score=${row.score.toFixed(3)})`);
        document.getElementById('statusBadge').className='badge red';
        document.getElementById('statusBadge').innerText='ALERT';
      }
    });

  }catch(e){
    console.error(e);
  }
}


window.addEventListener('load', ()=>{
  createCharts();
  // warm-up fetches
  for(let i=0;i<4;i++) setTimeout(fetchLatest, i*200);
  // poll regularly
  setInterval(fetchLatest, 1000);
});

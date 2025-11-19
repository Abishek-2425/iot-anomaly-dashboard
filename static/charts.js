// static/charts.js - Clean Chart.js logic for real-time simulation (Top 5 devices)

let tempChart, humChart, batChart;
let labels = [];
let tempData = [], humData = [], batData = [];
let deviceStatus = {};

// Active alerts tracking
let activeAlerts = {};  // key = device, value = alert div


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

// update device table start

function updateDeviceTable(device, t, h, b, status){
  // Save current device state
  deviceStatus[device] = {t,h,b,status};

  const tbody = document.querySelector('#deviceTable tbody');
  tbody.innerHTML = '';

  // Convert deviceStatus to array
  let devicesArray = Object.entries(deviceStatus);

  // Shuffle array slightly to rotate non-anomalies
  const anomalies = devicesArray.filter(([_, r]) => r.status);
  let normals = devicesArray.filter(([_, r]) => !r.status);

  // Shuffle normals
  for (let i = normals.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [normals[i], normals[j]] = [normals[j], normals[i]];
  }

  // Merge anomalies on top
  const sortedDevices = [...anomalies, ...normals].slice(0, 5);

  // Populate table
  sortedDevices.forEach(([d, r]) => {
    const statusHtml = r.status 
      ? ('<span class="status-dot dot-red"></span><strong style="color:#ef4444">ANOMALY</strong>') 
      : ('<span class="status-dot dot-green"></span>Normal');

    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${d}</td>
                    <td>${typeof r.t === 'number' ? r.t.toFixed(2) : r.t}</td>
                    <td>${typeof r.h === 'number' ? r.h.toFixed(2) : r.h}</td>
                    <td>${typeof r.b === 'number' ? r.b.toFixed(2) : r.b}</td>
                    <td>${statusHtml}</td>`;
    tbody.appendChild(tr);
  });
}


function pushAlert(device, msg) {
  const alerts = document.getElementById('alerts');

  // If this device already has an active alert, update timestamp/message instead of adding new
  if (activeAlerts[device]) {
    activeAlerts[device].innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    return;
  }

  // Create new alert div
  const div = document.createElement('div');
  div.className = 'alert';
  div.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
  
  alerts.prepend(div);
  activeAlerts[device] = div;

  // Limit max 8 alerts
  while (alerts.children.length > 8) {
    const lastChild = alerts.lastChild;
    // Remove from activeAlerts if present
    Object.keys(activeAlerts).forEach(d => {
      if (activeAlerts[d] === lastChild) delete activeAlerts[d];
    });
    alerts.removeChild(lastChild);
  }
}


// Optimized fetchLatest
async function fetchLatest(){
  try {
    const res = await fetch('/data/latest');
    const j = await res.json();
    if(!j.success){ console.error('No data', j); return; }

    j.rows.forEach(row => {
      const device = row.Device_ID || row.device || ("dev_"+Math.floor(Math.random()*999));
      const keys = Object.keys(row);
      const tKey = keys.find(k=>/temp/i.test(k)) || keys.find(k=>/temperature/i.test(k));
      const hKey = keys.find(k=>/hum/i.test(k)) || keys.find(k=>/humidity/i.test(k));
      const bKey = keys.find(k=>/bat/i.test(k)) || keys.find(k=>/battery/i.test(k));
      const t = tKey ? parseFloat(row[tKey]) : 0;
      const h = hKey ? parseFloat(row[hKey]) : 0;
      const b = bKey ? parseFloat(row[bKey]) : 0;
      const isAnom = row.anomaly ? 1 : 0;

      updateCharts(t,h,b);
      updateDeviceTable(device, t, h, b, isAnom);

      // Handle alerts
      const statusBadge = document.getElementById('statusBadge');
      if(isAnom){
        pushAlert(device, `${device} flagged as anomaly (score=${row.score !== null ? row.score.toFixed(3) : 'n/a'})`);
        statusBadge.className='badge red';
        statusBadge.innerText='ALERT';
      } else {
        statusBadge.className='badge green';
        statusBadge.innerText='LIVE';
      }
    });
  } catch(e){
    console.error(e);
  }
}

// fatch latest end

window.addEventListener('load', ()=>{
  createCharts();
  // warm-up fetches
  for(let i=0;i<4;i++) setTimeout(fetchLatest, i*200);
  // poll regularly
  setInterval(fetchLatest, 1000);
});

const API_URL = window.location.origin;

const form      = document.getElementById('predictForm');
const submitBtn = document.getElementById('submitBtn');
const resultCard    = document.getElementById('resultCard');
const resultValue   = document.getElementById('resultValue');
const resultExpl    = document.getElementById('resultExplanation');
const errorMsg      = document.getElementById('errorMsg');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorMsg.classList.remove('visible');
    resultCard.classList.remove('visible');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Predicting…';

    const payload = {
        active_users:    parseFloat(document.getElementById('active_users').value),
        time_of_day:     parseFloat(document.getElementById('time_of_day').value),
        background_jobs: parseFloat(document.getElementById('background_jobs').value),
        db_latency_ms:   parseFloat(document.getElementById('db_latency_ms').value),
    };

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const detail = await res.json();
            throw new Error(detail.detail || `Server error ${res.status}`);
        }

        const data = await res.json();
        const pct  = data.cpu_usage_pct;

        // colour-code the result
        let cls = 'low';
        if (pct > 75)      cls = 'high';
        else if (pct > 40) cls = 'mid';

        resultValue.className = 'result-value ' + cls;
        resultValue.innerHTML = `${pct.toFixed(1)} <span>%</span>`;

        // Build a human-readable explanation of the result
        let explanation = '';
        if (cls === 'low') {
            explanation = `With <strong>${payload.active_users}</strong> active users, <strong>${payload.background_jobs}</strong> background jobs, and <strong>${payload.db_latency_ms} ms</strong> database latency at hour <strong>${payload.time_of_day}</strong>, the server is under <strong>low load</strong>. CPU resources are largely available — no action needed.`;
        } else if (cls === 'mid') {
            explanation = `With <strong>${payload.active_users}</strong> active users, <strong>${payload.background_jobs}</strong> background jobs, and <strong>${payload.db_latency_ms} ms</strong> database latency at hour <strong>${payload.time_of_day}</strong>, the server is experiencing <strong>moderate load</strong>. Consider monitoring trends — usage may climb if traffic increases.`;
        } else {
            explanation = `With <strong>${payload.active_users}</strong> active users, <strong>${payload.background_jobs}</strong> background jobs, and <strong>${payload.db_latency_ms} ms</strong> database latency at hour <strong>${payload.time_of_day}</strong>, the server is under <strong>heavy load</strong>. Consider scaling resources or reducing background tasks to prevent degradation.`;
        }
        resultExpl.className = 'result-explanation ' + cls;
        resultExpl.innerHTML = explanation;

        resultCard.classList.add('visible');
    } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.classList.add('visible');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Predict CPU Usage';
    }
});

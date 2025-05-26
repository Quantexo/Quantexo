document.addEventListener('DOMContentLoaded', function() {
    const sectorSelect = document.getElementById('sector');
    const companySelect = document.getElementById('company');
    const symbolInput = document.getElementById('symbol');
    const searchBtn = document.getElementById('search');
    const statusDiv = document.getElementById('status');
    const chartDiv = document.getElementById('chart');

    // Fetch sector/company mapping
    fetch('/sectors')
        .then(res => res.json())
        .then(data => {
            sectorSelect.innerHTML = '<option value="">Select Sector</option>';
            Object.keys(data).forEach(sector => {
                sectorSelect.innerHTML += `<option value="${sector}">${sector}</option>`;
            });
        });

    sectorSelect.addEventListener('change', function() {
        const sector = sectorSelect.value;
        companySelect.innerHTML = '<option value="">Select Company</option>';
        if (sector) {
            fetch('/sectors')
                .then(res => res.json())
                .then(data => {
                    (data[sector] || []).forEach(company => {
                        companySelect.innerHTML += `<option value="${company}">${company}</option>`;
                    });
                });
        }
    });

    searchBtn.addEventListener('click', function() {
        let symbol = symbolInput.value.trim();
        if (!symbol && companySelect.value) symbol = companySelect.value;
        if (!symbol) {
            statusDiv.textContent = "⚠️ Please enter or select a company.";
            chartDiv.innerHTML = "";
            return;
        }
        statusDiv.textContent = "🔎 Analyzing " + symbol + "...";
        chartDiv.innerHTML = "";
        fetch('/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol})
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                statusDiv.textContent = data.error;
                chartDiv.innerHTML = "";
            } else {
                statusDiv.textContent = "";
                Plotly.newPlot('chart', data.traces, data.layout, {responsive: true});
            }
        })
        .catch(err => {
            statusDiv.textContent = "Error: " + err;
            chartDiv.innerHTML = "";
        });
    });
});
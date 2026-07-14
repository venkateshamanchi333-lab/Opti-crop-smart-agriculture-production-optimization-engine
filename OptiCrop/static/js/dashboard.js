document.addEventListener('DOMContentLoaded', () => {
    const charts = document.querySelectorAll('canvas[data-chart]');
    charts.forEach(canvas => {
        const chartType = canvas.dataset.chart;
        const labels = JSON.parse(canvas.dataset.labels || '[]');
        const values = JSON.parse(canvas.dataset.values || '[]');
        const config = {
            type: chartType,
            data: {
                labels,
                datasets: [{
                    label: canvas.dataset.label || 'Dataset',
                    data: values,
                    backgroundColor: ['#4eb77a', '#7ce392', '#71b584', '#3f7c56', '#2b5137'],
                    borderColor: '#ffffff',
                    borderWidth: 1,
                    fill: chartType !== 'bar',
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: chartType !== 'doughnut' },
                    tooltip: { mode: 'index', intersect: false }
                }
            }
        };
        new Chart(canvas, config);
    });
});

const rangeInputs = document.querySelectorAll('input[type=range]');
const sliders = document.querySelectorAll('.slider-value');

const navigationEntry = performance.getEntriesByType('navigation')[0];
if (navigationEntry && navigationEntry.type === 'reload' && window.location.pathname !== '/') {
    window.location.replace('/');
}

if (rangeInputs.length && sliders.length) {
    rangeInputs.forEach((input, idx) => {
        const output = sliders[idx];
        const update = () => { output.textContent = input.value; };
        input.addEventListener('input', update);
        update();
    });
}

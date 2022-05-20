
var mindist_slider = document.getElementById("mindist-slider")
var minheight_slider = document.getElementById("minheight-slider")
var mindist_valuebox = document.getElementById("mindist-value")
var minheight_valuebox = document.getElementById("minheight-value")
var freq_number = document.getElementById("frequency-number")
var tmax_number = document.getElementById("duration-number")

if (window.innerWidth < 600) {
    tmax_number.value = 3.0;
}
var mindist, minheight, tmax, freq, fs = 30;
var ts = [], ys = [], ymin, ymax;

mulberry_seed(0);

function create_data() {
    ts = [], ys = [];
    for (let t = 0; t <= tmax; t += 1.0/fs) {
        let y = Math.sin(2*Math.PI*freq * t) + uniform_sum_distribution(0, 0.5, 5);
        ts.push(t);
        ys.push(y);
    }
    ymin = Math.min(...ys), ymax = Math.max(...ys);
}

function update_chart() {
    let container = document.getElementById("findpeaks-playground-plot")

    let maxima = find_local_maxima(ys);
    let maxima_ts = maxima.map(i => ts[i]);
    let peak_indices = filter_maxima(maxima, ys, mindist*fs, minheight);
    let peak_ts = peak_indices.map(i => ts[i]);
    let peak_ys = peak_indices.map(i => ys[i]);
    let peak_indices_sorted = argsort(peak_ys);
    let max_t = peak_ts[peak_indices_sorted[peak_ts.length-1]]

    let data = [{
        x: ts,
        y: ys,
        name: "Raw signal",
        hoverinfo: "skip",
    }, {
        x: peak_ts,
        y: peak_ys,
        name: "Filtered peaks",
        mode: "markers",
        hoverinfo: "skip",
    }, {
        x: [ts[0], ts[ts.length-1]],
        y: [minheight, minheight],
        name: "Minimum height",
        mode: "lines",
        line: {
            dash: "dot",
            width: 3,
        },
        hovertemplate: minheight.toFixed(2),
    }, {
        x: [max_t, max_t+mindist],
        y: [minheight, minheight],
        name: "Minimum dist.",
        line: {
            width: 4,
        },
        marker: {
            size: 8,
        },
        hovertemplate: mindist.toFixed(2) + " s",
    }];
    for (let i=0; i < maxima.length; ++i) {
        data.push({
            x: [maxima_ts[i], maxima_ts[i]],
            y: [ymin, ymax],
            name: "Local maxima",
            mode: "lines",
            line: {
                color: "rgba(51,51,51, 0.3)",
                dash: "dash"
            },
            hoverinfo: "skip",
            showlegend: i==0,
        })
    }

    Plotly.newPlot(
        container,
        data,
        {
            template: plotlyDefault,
            showlegend: true,
            xaxis: {
                title: "Time / s",
                range: [ts[0]-0.05, ts[ts.length-1]+0.05],
            },
            yaxis: {
                title: "Amplitude",
            }
        },
        {
            responsive: false,
            displayModeBar: false,
        }
    );
}
function parse_sliders() {
    mindist = parseFloat(mindist_slider.value) / mindist_slider.max * ts[ts.length-1] / 2;
    mindist_valuebox.innerText = mindist.toFixed(2) + " s";

    minheight = (parseFloat(minheight_slider.value)
                 / (parseFloat(minheight_slider.max)+1)
                 * (ymax-ymin) + ymin);
    minheight_valuebox.innerText = minheight.toFixed(2);
}
function sliders_changed() {
    parse_sliders();
    update_chart();
}
mindist_slider.oninput = sliders_changed;
minheight_slider.oninput = sliders_changed;

document.body.onresize = update_chart;

function refresh_data() {
    freq = parseFloat(freq_number.value);
    tmax = parseFloat(tmax_number.value);

    create_data();
    parse_sliders();

    update_chart();
}
refresh_data();

console.log(argsort([80, 40, 10, 60, 20]))

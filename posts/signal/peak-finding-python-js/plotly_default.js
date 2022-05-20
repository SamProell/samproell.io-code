var baseLayout = {
    font: {
        family: 'arial, sans-serif',
        size: 14,
    },
    xaxis: {
        automargin: true,
        showline: true,
        zeroline: false,
        linewidth: 2,
        ticks: "outside",
        tickwidth: 2,
        showgrid: false,
    },
    yaxis: {
        automargin: true,
        showline: true,
        zeroline: false,
        linewidth: 2,
        ticks: "outside",
        tickwidth: 2,
        showgrid: false,
    },
    showlegend: false,
    legend: {
        x: 1,
        y: 1,
        xanchor: 'right',
        yanchor: "bottom",
        orientation: "h",
    },
    margin: {
        autoexpand: true,
        t: 4, b: 4, l: 4, r: 4,
    }
}
var plotlyDefault = {
    data: {}, layout: baseLayout
}

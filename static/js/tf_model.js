let scalerMean, scalerStd;

async function loadCSVData() {
    const response = await fetch('static/ad_mini.csv');
    const data = await response.text();
    const lines = data.trim().split('\n');
    lines.shift();
    const samples = [];
    const labels = [];

    lines.forEach(line => {
        const cols = line.split(',');
        const timeSpent = parseFloat(cols[0]);
        const age = parseFloat(cols[1]);
        const areaIncome = parseFloat(cols[2]);
        const internetUsage = parseFloat(cols[3]);
        const gender = (cols[4].trim().toLowerCase() === 'male') ? 0 : 1;
        const clicked = parseFloat(cols[5]);
        samples.push([timeSpent, age, areaIncome, internetUsage, gender]);
        labels.push(clicked);
    });
    return { samples, labels };
}

async function trainAndSaveTFModel() {

    const { samples, labels } = await loadCSVData();
    const xs = tf.tensor2d(samples);
    const ys = tf.tensor2d(labels, [labels.length, 1]);

    const moments = tf.moments(xs, 0);
    scalerMean = moments.mean;
    scalerStd = tf.sqrt(moments.variance);
    const xsStandardized = xs.sub(scalerMean).div(scalerStd);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [5] }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    console.log('Entraînement du modèle TF.js...');
    await model.fit(xsStandardized, ys, { epochs: 10, batchSize: 32 });
    console.log('Entraînement terminé, sauvegarde dans localStorage…');
    await model.save('localstorage://tfjs-model');
    console.log('Modèle TF.js sauvegardé dans localStorage.');

    window.scalerMean = scalerMean;
    window.scalerStd = scalerStd;
    return model;
}

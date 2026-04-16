// Import necessary modules
const { Worker, isMainThread, parentPort } = require('worker_threads');

// Define the transformer task
const transformerTask = (data) => {
    // Implement your transformation logic here
    return data.map(item => ({
        ...item,
        transformed: true // Example transformation
    }));
};

// Main thread execution
if (isMainThread) {
    const dataChunks = [/* Your data array to transform */];
    const numWorkers = require('os').cpus().length; // Get the number of CPU cores
    const workerPromises = [];

    // Split data into chunks for each worker
    const chunkSize = Math.ceil(dataChunks.length / numWorkers);
    for (let i = 0; i < numWorkers; i++) {
        const workerData = dataChunks.slice(i * chunkSize, (i + 1) * chunkSize);
        workerPromises.push(new Promise((resolve, reject) => {
            const worker = new Worker(__filename);
            worker.on('message', resolve);
            worker.on('error', reject);
            worker.on('exit', (code) => {
                if (code !== 0) reject(new Error(`Worker stopped with exit code ${code}`));
            });
            worker.postMessage(workerData);
        }));
    }

    // Execute all workers and collect results
    Promise.all(workerPromises)
        .then(results => {
            // Combine results from all workers
            const finalResult = results.flat();
            console.log('Final transformed data:', finalResult);
        })
        .catch(err => console.error('Error:', err));
}

// Worker thread execution
else {
    parentPort.on('message', (workerData) => {
        const transformedData = transformerTask(workerData);
        parentPort.postMessage(transformedData);
    });
}
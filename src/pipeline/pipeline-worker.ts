import { Worker } from 'worker_threads';

// Define worker data type
interface WorkerData {
    stage: string;
    data: any;
}

// Function to start a worker thread
function startWorker(workerData: WorkerData) {
    return new Promise((resolve, reject) => {
        const worker = new Worker('./worker.js', { workerData });
        worker.on('message', resolve);
        worker.on('error', reject);
        worker.on('exit', (code) => {
            if (code !== 0)
                reject(new Error(`Worker stopped with exit code ${code}`));
        });
    });
}

export default startWorker;

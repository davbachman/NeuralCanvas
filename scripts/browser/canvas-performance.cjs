const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs/promises');
(async () => {
    const label = process.argv[2] || 'baseline';
    const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || '/usr/bin/chromium', args: ['--no-sandbox'] });
    const results = [];
    for (const fixture of ['onehot', 'embedding'])
        for (const rate of [1, 4]) {
            const context = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
            const page = await context.newPage();
            const client = await context.newCDPSession(page);
            await client.send('Performance.enable');
            await page.goto(process.env.CANVAS_URL || 'http://localhost:4173/NeuralCanvas/');
            await page.getByLabel('Import state file').setInputFiles(`output/canvas-${fixture}.json`);
            await page.waitForTimeout(3500);
            if (await page.getByRole('alert').count())
                throw Error(await page.getByRole('alert').innerText());
            if (await page.locator('.react-flow__node').count() < 15)
                throw Error('Missing fixture nodes');
            await client.send('Emulation.setCPUThrottlingRate', { rate });
            await page.evaluate(() => {
                window.perf = { frames: [], long: [], json: [], puts: [] };
                let last = performance.now();
                function frame(now) { window.perf.frames.push(now - last); last = now; requestAnimationFrame(frame); }
                requestAnimationFrame(frame);
                new PerformanceObserver(list => { window.perf.long.push(...list.getEntries().map(e => e.duration)); }).observe({ type: 'longtask' });
                for (const method of ['parse', 'stringify']) {
                    const original = JSON[method];
                    JSON[method] = function (...args) { const start = performance.now(); const result = original.apply(this, args); const size = method === 'parse' ? args[0]?.length : result?.length; if (size > 50000)
                        window.perf.json.push({ method, ms: performance.now() - start, size }); return result; };
                }
                const put = IDBObjectStore.prototype.put;
                IDBObjectStore.prototype.put = function (...args) { const start = performance.now(); const result = put.apply(this, args); window.perf.puts.push(performance.now() - start); return result; };
            });
            const box = await page.locator('.react-flow').boundingBox();
            const x = box.x + box.width * .7, y = box.y + box.height * .8;
            for (const mode of ['pan', 'zoom']) {
                await page.evaluate(() => { for (const key of Object.keys(window.perf))
                    window.perf[key] = []; });
                await client.send('Tracing.start', { categories: 'devtools.timeline,v8,disabled-by-default-v8.cpu_profiler', options: 'sampling-frequency=1000', transferMode: 'ReturnAsStream' });
                const before = (await client.send('Performance.getMetrics')).metrics;
                const start = Date.now();
                if (mode === 'pan') {
                    // Secondary button pans even when the pointer starts over a card.
                    for (let round = 0; round < 6; round++) {
                        await page.mouse.move(x, y);
                        await page.mouse.down({ button: 'right' });
                        for (let i = 1; i <= 25; i++) {
                            await page.mouse.move(x + Math.sin(i / 25 * Math.PI * 2) * 120, y + Math.cos(i / 25 * Math.PI * 2) * 60);
                            await page.waitForTimeout(16);
                        }
                        await page.mouse.up({ button: 'right' });
                        await page.waitForTimeout(100);
                    }
                }
                else {
                    await page.mouse.move(x, y);
                    for (let i = 0; i < 150; i++) {
                        await page.mouse.wheel(0, i % 50 < 25 ? -12 : 12);
                        await page.waitForTimeout(30);
                    }
                }
                await page.waitForTimeout(500);
                const elapsed = Date.now() - start;
                const after = (await client.send('Performance.getMetrics')).metrics;
                const done = new Promise(resolve => client.once('Tracing.tracingComplete', resolve));
                await client.send('Tracing.end');
                const { stream } = await done;
                let raw = '';
                while (true) {
                    const r = await client.send('IO.read', { handle: stream });
                    raw += r.data;
                    if (r.eof)
                        break;
                }
                await client.send('IO.close', { handle: stream });
                await fs.writeFile(`output/${label}-${fixture}-${rate}-${mode}.trace.json`, raw);
                const trace = JSON.parse(raw).traceEvents;
                const spans = trace.filter(e => e.ph === 'X' && e.dur);
                const sum = names => spans.filter(e => names.includes(e.name)).reduce((n, e) => n + e.dur / 1000, 0);
                const measures = await page.evaluate(() => window.perf);
                const frames = measures.frames.sort((a, b) => a - b);
                const delta = name => 1000 * (after.find(m => m.name === name).value - before.find(m => m.name === name).value);
                results.push({ fixture, rate, mode, elapsed, frames: frames.length, frameP95: frames[Math.floor(frames.length * .95)], framesOver50: frames.filter(x => x > 50).length, longTasks: measures.long.length, longTaskMs: measures.long.reduce((a, b) => a + b, 0), scriptMs: delta('ScriptDuration'), layoutMs: delta('LayoutDuration'), styleMs: delta('RecalcStyleDuration'), taskMs: delta('TaskDuration'), paintMs: sum(['Paint']), gcMs: sum(['MinorGC', 'MajorGC']), jsonMs: measures.json.reduce((s, e) => s + e.ms, 0), largeJsonCalls: measures.json.length, putMs: measures.puts.reduce((a, b) => a + b, 0), writes: measures.puts.length, heapMB: after.find(m => m.name === 'JSHeapUsedSize').value / 1e6 });
                console.log(JSON.stringify(results.at(-1)));
            }
            await context.close();
        }
    await fs.writeFile(`output/${label}-canvas-perf.json`, JSON.stringify(results, null, 2));
    await browser.close();
})().catch(e => { console.error(e); process.exit(1); });

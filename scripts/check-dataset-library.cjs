const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {createRequire}=require('node:module');
const {chromium}=createRequire(path.resolve(process.env.PLAYWRIGHT_ROOT || '.', 'package.json'))('playwright');
const root=path.resolve(__dirname, '..');
const output=path.resolve(process.env.OUTPUT_DIR || 'output/dataset-library');fs.mkdirSync(output,{recursive:true});
const {execFileSync}=require('node:child_process');
(async()=>{
 const b=await chromium.launch({headless:true,executablePath:'/usr/bin/chromium',args:['--no-sandbox']});
 const errors=[],reports=[];
 try{
  const p=await b.newPage({viewport:{width:1600,height:1100},acceptDownloads:true});p.setDefaultTimeout(45000);p.on('pageerror',e=>errors.push(e.message));
  const model=JSON.parse(fs.readFileSync(path.join(root,'public/models/linear.json')));
  // A familiar saved project gives us a real Dataset block and its existing connections.
  await p.goto(process.env.APP_URL || 'http://127.0.0.1:5195/NeuralCanvas/');
  await p.getByLabel('Import state file').setInputFiles({name:'football-model.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(model))});
  const dataset=model.state.graph.nodes.find(n=>n.type==='dataset');
  await p.locator(`[data-id="${dataset.id}"]`).dispatchEvent('click');
  await p.getByRole('tab',{name:'Details',exact:true}).click();
  const cardSource=p.getByRole('combobox',{name:`Dataset for ${dataset.label}`,exact:true});
  const shortcuts=['line-1d','cubic-1d','plane-2d','threshold-1d','circle-center','parabola-boundary','neuron-basics','xor'];
  assert.deepEqual(await cardSource.locator('option:not([hidden])').evaluateAll(options=>options.map(option=>option.value)),[...shortcuts,'browse']);
  for(const kind of shortcuts){
   const label=await cardSource.locator(`option[value="${kind}"]`).textContent();
   await cardSource.selectOption(kind);
   assert.equal(await cardSource.locator('option:checked').textContent(),label);
   assert.equal(await p.getByRole('dialog').count(),0);
  }
  reports.push({workflow:'direct card selection of all eight simple regression and classification datasets',passed:true});
  await cardSource.selectOption('browse');
  const dialog=p.getByRole('dialog',{name:'Dataset library'});
  await dialog.getByRole('button',{name:/Football stories/}).click();
  await dialog.getByRole('button',{name:'Use this dataset',exact:true}).waitFor();
  await p.waitForFunction(()=>!document.querySelector('.dataset-library-use')?.disabled);
  await p.screenshot({path:path.join(output,'football.png')});
  await dialog.getByRole('button',{name:'Mixed training · football benchmark',exact:true}).click();
  await dialog.getByText('3,840',{exact:true}).waitFor();
  assert.equal(await dialog.getByLabel('Question selection',{exact:true}).inputValue(),'benchmark');
  await dialog.getByLabel('Football model inputs',{exact:true}).selectOption('answer-token');
  await dialog.getByText('[23]',{exact:true}).first().waitFor();
  await p.waitForFunction(()=>!document.querySelector('.dataset-library-use')?.disabled);
  await p.screenshot({path:path.join(output,'answer-token.png')});
  await dialog.getByRole('button',{name:'Use this dataset',exact:true}).click();
  await dialog.waitFor({state:'hidden'});
  assert.equal(await p.getByRole('button',{name:'Replace text dataset',exact:true}).count(),1);
  await p.getByRole('button',{name:'File',exact:true}).click();const download=p.waitForEvent('download');await p.getByRole('menuitem',{name:'Save',exact:true}).click();
  const saved=JSON.parse(fs.readFileSync(await(await download).path()));
  assert.equal(saved.state.graph.nodes.find(n=>n.id===dataset.id).params.textData.maxLength,23);
  assert.equal(saved.state.graph.nodes.find(n=>n.id===dataset.id).params.datasetLibraryPreset,'football-qa');
  reports.push({workflow:'football answer-token selection and saved project',passed:true});
  await p.getByLabel('Import state file').setInputFiles({name:'saved.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(saved))});
  await p.locator(`[data-id="${dataset.id}"]`).dispatchEvent('click');await p.getByRole('tab',{name:'Details',exact:true}).click();
  await p.getByRole('button',{name:'Browse datasets…',exact:true}).click();
  await dialog.getByRole('button',{name:'Regression',exact:true}).click();
  await p.screenshot({path:path.join(output,'regression.png')});
  await dialog.getByRole('button',{name:/California housing/}).click();
  await dialog.getByText('16,346',{exact:true}).waitFor();
  await p.screenshot({path:path.join(output,'housing.png')});
  await dialog.getByRole('button',{name:'Close dataset library',exact:true}).click();
  await p.getByRole('button',{name:'Browse datasets…',exact:true}).click();
  await p.screenshot({path:path.join(output,'all-datasets.png')});
  await dialog.getByRole('button',{name:'Close dataset library',exact:true}).click();
  // The existing source-selector route still imports each prepared JSON with default controls.
  for(const filename of ['imdb.json','football-counts.json']){
   await p.getByRole('combobox',{name:'Dataset selection',exact:true}).selectOption('custom-text');
   await p.getByLabel('Choose text data file',{exact:true}).setInputFiles(path.join(root,'public/datasets',filename));
   await p.getByRole('button',{name:'Import text dataset',exact:true}).click();
   await p.getByRole('dialog',{name:'Import text data'}).waitFor({state:'hidden'});
   reports.push({workflow:'existing prepared JSON import',filename,passed:true});
  }
  await p.getByRole('button',{name:'Browse datasets…',exact:true}).click();
  await dialog.getByRole('button',{name:'Use a custom dataset…',exact:true}).click();
  await dialog.getByRole('button',{name:'Import prepared JSON or text…',exact:true}).click();
  await p.getByLabel('Choose text data file',{exact:true}).setInputFiles(path.join(root,'public/datasets/football-counts.json'));
  await p.getByRole('button',{name:'Import text dataset',exact:true}).click();await p.getByRole('dialog',{name:'Import text data'}).waitFor({state:'hidden'});
  reports.push({workflow:'library custom JSON import',passed:true});
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(output,'browser.json'),JSON.stringify({browser:await b.version(),appBase:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),reports,errors},null,2));
  console.log('Real browser library, prepared JSON imports, persistence and preview checks passed.');
 }finally{await b.close();}
})().catch(e=>{console.error(e);process.exitCode=1});

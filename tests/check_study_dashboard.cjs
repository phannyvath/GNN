const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(require('path').join(__dirname,'../results/comparison/dashboard.html'),'utf8');
const data=JSON.parse(html.split('<script id="study-data" type="application/json">')[1].split('</script>')[0]);
class Element{constructor(id){this.value=id==='dataset'?'THEIA':id==='metric'?'f1':id==='priority'?'recall':'';this.textContent=id==='study-data'?JSON.stringify(data):'';this._html='';}set innerHTML(v){this._html=v;const first=v.match(/<option value="([^"]*)"/);if(first)this.value=first[1];}get innerHTML(){return this._html;}querySelectorAll(){return [];}scrollIntoView(){}click(){}}
const elements={},get=id=>elements[id]??(elements[id]=new Element(id));
const ctx={document:{getElementById:get,createElement:()=>new Element('a')},console,Blob,URL,setTimeout,window:{print(){}}};vm.createContext(ctx);vm.runInContext(html.split('<script>')[1].split('</script>')[0],ctx);
assert.equal(get('run').value,'study_full_v1');assert(get('metrics').innerHTML.includes('0.863'));
for(const run of data.runs){get('dataset').value=run.dataset;vm.runInContext(`chooseRuns(${JSON.stringify(run.id)})`,ctx);assert(get('metrics').innerHTML.includes('table'));for(const seed of new Set(run.metrics.map(r=>r.seed))){get('seed').value=String(seed);get('seed').onchange();assert(get('rowcount').textContent.startsWith(String(run.metrics.filter(r=>r.seed===seed&&['flash_node','flash_inspired','magic_node','magic_inspired'].includes(r.model)).length)));}for(const metric of ['f1','recall','false_positive_rate']){get('metric').value=metric;get('metric').onchange();assert(!get('chart').innerHTML.includes('NaN'));}}
get('search').value='SYNTHETIC';get('search').oninput();assert(!get('archiveTable').innerHTML.includes('study_full_v1'));get('export').onclick();console.log('Passed: initialization, 13 experiment selections, every seed, metric switches, archive search and CSV export handler (DOM harness; not browser layout testing).');

get('dataset').value='THEIA';vm.runInContext("chooseRuns('study_full_v1')",ctx);
assert(get('datasetCounts').innerHTML.includes('1,623,966'));
assert(get('recommendations').innerHTML.includes('Random Forest'));
for(const priority of ['recall','false_positive_rate','inference_seconds']){get('priority').value=priority;get('priority').onchange();assert(get('recommendations').innerHTML.includes('Your practical priority'));}
vm.runInContext("chooseRuns('study_smoke_v1')",ctx);assert(get('recommendations').innerHTML.includes('Insufficient evidence'));assert(!get('recommendations').innerHTML.includes('Your practical priority'));
vm.runInContext("current={id:'empty',dataset:'THEIA',unit:'nodes',status:'Incomplete',metrics:[],config:{},artifacts:{},figures:[],evidence:{run_state:'queued',compatibility_reason:'Insufficient evidence'}};render()",ctx);assert(get('plain').innerHTML.includes('queued'));assert(get('cards').innerHTML.includes('Unavailable'));
console.log('Passed recommendation preferences, incomplete-run gating, empty queued snapshot and exact dataset count.');

get('dataset').value='THEIA';vm.runInContext("chooseRuns('study_full_v1')",ctx);
assert(get('chart').innerHTML.includes('FLASH-inspired'));
assert(get('chart').innerHTML.includes('MAGIC-inspired'));
assert(!get('chart').innerHTML.includes('Random Forest'));
assert(!get('metrics').innerHTML.includes('Random Forest'));
assert(get('baselineTable').innerHTML.includes('Random Forest'));
assert(get('baselineTable').innerHTML.includes('Primary method'));
assert(get('baselineTable').innerHTML.includes('0.947'));
console.log('Passed primary-method focus and separate measured baseline table.');

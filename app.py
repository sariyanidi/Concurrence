from flask import Flask, request, render_template_string, jsonify, Response, stream_with_context
import os
import argparse 
import subprocess
import threading

parser = argparse.ArgumentParser()
parser.add_argument("port", type=int, nargs='?', default=8000)
args = parser.parse_args()

app = Flask(__name__)
current_process = None
process_lock = threading.Lock()

HTML = """
<!doctype html>
<html>
  <head>
    <style>
      /* Base styles */
      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f0f2f5;
        color: #333;
        margin: 0;
        padding: 2rem;
      }
      /* Controls and labels */
      .controls {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.5rem;
        align-items: center;
      }
      .controls label {
        display: inline-block;
        width: 220px;
        font-weight: 600;
      }
      .controls input[type="text"],
      .controls input[type="number"] {
        flex: 1;
        padding: 0.5rem 1rem;
        border: 1px solid #ccc;
        border-radius: 6px;
        font-size: 0.9rem;
        background: #fff;
      }
      .controls button {
        padding: 0.5rem 1.2rem;
        border: none;
        border-radius: 6px;
        background-color: #4a90e2;
        color: white;
        font-size: 0.9rem;
        cursor: pointer;
        transition: background-color 0.2s;
      }
      .controls button:hover {
        background-color: #357ab8;
      }
      /* Action buttons */
      .buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.5rem;
      }
      .buttons button {
        background-color: #4a90e2;
        color: #fff;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.4rem;
        cursor: pointer;
        transition: background-color 0.2s;
      }
      .buttons button:hover {
        background-color: #357ab8;
      }
      /* Panels grid */
      .panels {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
      }
      .panel {
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        padding: 1rem;
        max-height: 300px;
        overflow-y: auto;
      }
      .panel h4 {
        margin: 0 0 0.5rem;
        font-size: 1.1rem;
        color: #4a90e2;
      }
      .panel div {
        font-size: 0.9rem;
        line-height: 1.4;
      }
      /* Script output console */
      #script-output {
        background-color: #1e1e1e;
        color: #d4d4d4;
        border-radius: 6px;
        padding: 1rem;
        font-family: Menlo, Consolas, monospace;
        font-size: 0.85rem;
        max-height: 500 px;
        overflow-y: auto;   
        white-space: pre-wrap;
      }
      .tooltip-icon {
        display: inline-block;
        margin-left: 4px;
        width: 16px;
        height: 16px;
        background-color: #4a90e2;
        color: white;
        border-radius: 50%;
        text-align: center;
        font-size: 12px;
        line-height: 16px;
      }
      .tooltip-icon:hover {
        background-color: #357ab8;
      }
      
      /* Copy output button */
      #copy-cmd {
        padding: 0.5rem 1.2rem;
        border: none;
        border-radius: 6px;
        background-color: #50c878;
        color: white;
        cursor: pointer;
        transition: background-color 0.2s;
      }
      #copy-cmd:hover {
        background-color: #3ba35f;
      }
      .clearfix::after {
      content: "";
      display: table;
      clear: both;
    }
      .narrow-input input {
        width: 100px;
        flex: 0 0 auto;
      }
      .elegant-hr {
  border: none; border-top: 1px solid #ccc; margin: 2em 0;
}

    </style>
  </head>
  <body>
     <h3>Create file lists for training</h3>
    <!-- Controls -->
    <div class="controls">
      <label for="dir1">
      Dir with <i>x</i> files:
      </label>
      <input type="text" id="dir1" placeholder="/path/to/dir/with/x/signal/files" value="" />
      <button id="load1">Load Dir</button>
    </div>
    <div class="controls">
      <label for="dir2">Dir with <i>y</i> files</label>
      <input type="text" id="dir2" placeholder="/path/to/dir/with/y/signal/files" value="" />
      <button id="load2">Load Dir</button>
    </div>
    <div class="controls">
      <label for="filter1">Filter <i>x</i> files (Regex)
      <span class="tooltip-icon" title="You can apply regex filter to keep only files of signals. E.g., typing '^xsignal' will keep only files that start with 'xsignal', and 'signalx$' will keep only files ending in 'signalx'">i</span>
      </label>
      <input type="text" id="filter1" placeholder="regex pattern to filter x files" />
      <button id="apply1">Apply</button>
      <button id="clear1">Clear</button>
    </div>
    <div class="controls">
      <label for="filter2">Filter <i>y</i> files (Regex)
      <span class="tooltip-icon" title="You can apply regex filter to keep only files of signals. E.g., typing '^ysignal' will keep only files that start with 'ysignal', and 'signaly$' will keep only files ending in 'signaly'">i</span>
      </label>
      <input type="text" id="filter2" placeholder="regex pattern to filter y files" />
      <button id="apply2">Apply</button>
      <button id="clear2">Clear</button>
    </div>
    <div class="controls">
      <label for="seed">Shuffle Seed</label>
      <input type="number" id="seed" placeholder="optional" />
    </div>
    <div class="controls">
      <label for="out1"><i>x</i> files list will be written here</label>
      <input type="text" id="out1" value="xfiles_list.txt" />
      <label for="out2"><i>y</i> files list will be written here</label>
      <input type="text" id="out2" value="yfiles_list.txt" />
      <label for="out3">List of IDs of (<i>x</i>,<i>y</i>) pairs will be written here</label>
      <input type="text" id="out3" value="xy_file_ids.txt" />
    </div>

    <!-- Action Buttons -->
    <div class="buttons">
      <button id="compare">🔍 (1) Filter out unmatched pairs</button>
      <button id="compute-lcs">🧑‍🤝‍🧑 (2) Create pair IDs</button>
      <button id="shuffle">🔀 (3) Shuffle pairs (optional) </button>
    </div>

    <!-- Preview Panels -->
    <div class="panels">
      <div class="panel"><h4><i>x</i> files</h4><div id="list1"></div></div>
      <div class="panel"><h4><i>y</i> files</h4><div id="list2"></div></div>
      <div class="panel"><h4>File pair IDs</h4><div id="list3"></div></div>
    </div>
    <hr class="elegant_hr">
    <h3>Create file lists for testing (can skip for PSCS analyses)</h3>
        
    <div class="controls">
      <label for="pct">% of pairs for test:</label>
      <input type="number" id="pct" value="20" min="0" max="100" />
      <label for="bot1"><i>x</i> files list (test set) will be written here:</label>
      <input type="text" id="bot1" value="xfiles_list_tes.txt" />
      <label for="bot2"><i>y</i> files list (test set) will be written here:</label>
      <input type="text" id="bot2" value="yfiles_list_tes.txt" />
      <label for="bot3">List with IDs of <i>x</i>/<i>y</i> pair IDs (test set) will be written here:</label>
      <input type="text" id="bot3" value="xy_file_ids_tes.txt" />
    </div>
    
        <!-- Action Buttons -->
    <div class="buttons">
      <button id="split-bottom">🧑‍🤝‍🧑 Create test pairs</button>
      <!--<button id="save-lists">Save Lists</button>-->
    </div>

    
    <div class="panels">
      <div class="panel"><h4><i>x</i> files for test:</h4><div id="bottom1"></div></div>
      <div class="panel"><h4><i>y</i> files for test:</h4><div id="bottom2"></div></div>
      <div class="panel"><h4>File pair IDs for test</h4><div id="bottom3"></div></div>
    </div>
    
    <hr class="elegant_hr"/>
    
   <h3>Run Concurrence Script</h3>
    
    
    <div class="controls">
      <label for="segment_size">Segment size (w):</label>
      <input type="number" id="segment_size"  value="100" min="1" style="flex: 0 0 auto; width: 100px;" />
     </div>
     <div class="controls">
      <label for="PSCS_file">Filepath (csv) for PSCS values:</label>
      <input type="text" id="PSCS_file" style="flex: 1; width:100%;" min-width:600px; placeholder="(optional .csv file path)--PSCSs for each test pair will be computed and saved if filepath provided " />
      </div>
     <div class="controls">
      <label for="device-name">Device:</label>
      <input type="text" id="device-name" value="cuda" style="flex: 0 0 auto; width: 600px;" placeholder="cuda or cuda:0 etc. for NVIDIA GPU; cpu for CPU; or mps for Apple Silicon (m1, m2 etc.)" />
      </div>
     <div class="controls">
      <label for="extra-arguments">Extra arguments:</label>
      <input type="text" id="extra-arguments" value="" style="flex: 0 0 auto; width: 600px;" placeholder="Extra command line arguments. E.g., --segs_per_pair=5 will speed up code." />
      </div>
    
    <!-- Action Buttons -->
    <div class="buttons">
      <button id="show-cmd">🖥️ Show Cmd</button>
      <button id="copy-cmd">📋 Copy Cmd</button>
      <button id="run-script">⚙️ Run Cmd</button>
      <button id="abort-cmd">❌ Abort</button>

    </div>




    <!-- Script Output -->
    <h4>Script Output</h4>
    <div id="script-output"></div>
    <script>
    let abortController;
      // Element refs
      const dir1 = document.getElementById('dir1');
      const dir2 = document.getElementById('dir2');
      const filter1 = document.getElementById('filter1');
      const filter2 = document.getElementById('filter2');
      const pctInput = document.getElementById('pct');
      const seedInput = document.getElementById('seed');
      const segmentSizeInput = document.getElementById('segment_size');
      const deviceNameInput = document.getElementById('device-name');
      const extraArgumentsInput = document.getElementById('extra-arguments');
      const PSCSFileInput = document.getElementById('PSCS_file');
      const out1 = document.getElementById('out1');
      const out2 = document.getElementById('out2');
      const out3 = document.getElementById('out3');
      const bot1 = document.getElementById('bot1');
      const bot2 = document.getElementById('bot2');
      const bot3 = document.getElementById('bot3');
      const list1Container = document.getElementById('list1');
      const list2Container = document.getElementById('list2');
      const list3Container = document.getElementById('list3');
      const bottom1Container = document.getElementById('bottom1');
      const bottom2Container = document.getElementById('bottom2');
      const bottom3Container = document.getElementById('bottom3');
      const scriptOutput = document.getElementById('script-output');

      let list1 = [], list2 = [], list3 = [], bottom1 = [], bottom2 = [], bottom3 = [];

      async function loadList(path) {
        const res = await fetch('/scan', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ path }) });
        return await res.json();
      }
      function render(arr, container) { container.innerHTML = arr.length ? arr.map(x=>`<div>${x}</div>`).join('') : '<em>(none)</em>'; }
      function getBaseName(p) { return p.replace(/.*[\\\/]/,''); }
      function clearBottoms() { bottom1=[]; bottom2=[]; bottom3=[]; render(bottom1,bottom1Container); render(bottom2,bottom2Container); render(bottom3,bottom3Container); }
      function mulberry32(a) { return function(){ var t=a+=0x6D2B79F5; t=Math.imul(t^(t>>>15),t|1); t^=t+Math.imul(t^(t>>>7),t|61); return((t^(t>>>14))>>>0)/4294967296; }}

      document.getElementById('load1').onclick = async ()=>{list1=await loadList(dir1.value);render(list1,list1Container);clearBottoms();};
      document.getElementById('load2').onclick = async ()=>{list2=await loadList(dir2.value);render(list2,list2Container);clearBottoms();};
      document.getElementById('apply1').onclick = ()=>{const re=new RegExp(filter1.value||'$^');list1=list1.filter(p=>re.test(getBaseName(p)));render(list1,list1Container);clearBottoms();};
      document.getElementById('clear1').onclick = ()=>{filter1.value='';render(list1,list1Container);clearBottoms();};
      document.getElementById('apply2').onclick = ()=>{const re=new RegExp(filter2.value||'$^');list2=list2.filter(p=>re.test(getBaseName(p)));render(list2,list2Container);clearBottoms();};
      document.getElementById('clear2').onclick = ()=>{filter2.value='';render(list2,list2Container);clearBottoms();};

      document.getElementById('compare').onclick = ()=>{
        const rex1=filter1.value?new RegExp(filter1.value,'g'):null;
        const rex2=filter2.value?new RegExp(filter2.value,'g'):null;
        const norm1=list1.map(p=>{const b=getBaseName(p);return rex1?b.replace(rex1,''):b;});
        const norm2=list2.map(p=>{const b=getBaseName(p);return rex2?b.replace(rex2,''):b;});
        const inter=norm1.filter(n=>norm2.includes(n));
        list1=list1.filter((p,i)=>inter.includes(norm1[i])); list2=list2.filter((p,i)=>inter.includes(norm2[i]));
        render(list1,list1Container); render(list2,list2Container); clearBottoms();
      };

      document.getElementById('compute-lcs').onclick = ()=>{list3=[];const len=Math.min(list1.length,list2.length);for(let i=0;i<len;i++){const b1=getBaseName(list1[i]),b2=getBaseName(list2[i]);list3.push(longestCommonSubstr(b1,b2));}render(list3,list3Container);clearBottoms();};

      document.getElementById('shuffle').onclick = ()=>{
        const len=Math.min(list1.length,list2.length,list3.length);
        let rng=Math.random;const seed=parseInt(seedInput.value);
        if(!isNaN(seed)) rng=mulberry32(seed);
        const idx=Array.from({length:len},(_,i)=>i);
        for(let i=idx.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]];}
        list1=idx.map(i=>list1[i]);list2=idx.map(i=>list2[i]);list3=idx.map(i=>list3[i]);
        render(list1,list1Container);render(list2,list2Container);render(list3,list3Container);clearBottoms();
      };

      document.getElementById('split-bottom').onclick = async ()=>{
        const pctVal=parseFloat(pctInput.value),pct=isNaN(pctVal)?0.2:(pctVal/100);
        const i1=Math.floor(list1.length*(1-pct)),i2=Math.floor(list2.length*(1-pct)),i3=Math.floor(list3.length*(1-pct));
        bottom1=list1.slice(i1);bottom2=list2.slice(i2);bottom3=list3.slice(i3);
        list1=list1.slice(0,i1);list2=list2.slice(0,i2);list3=list3.slice(0,i3);
        render(list1,list1Container);render(list2,list2Container);render(list3,list3Container);
        render(bottom1,bottom1Container);render(bottom2,bottom2Container);render(bottom3,bottom3Container);
        await fetch('/split-bottom',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bottom1,bottom2,bottom3,b1:bot1.value.trim(),b2:bot2.value.trim(),b3:bot3.value.trim()})});
      };

      document.getElementById('run-script').onclick = async ()=>{
              if (abortController) abortController.abort();
        abortController = new AbortController();

              document.getElementById('abort-cmd').onclick = () => {
        if (abortController) {
          abortController.abort();
        }
        fetch('/abort', {method: 'POST'}).then(r => r.json()).then(j => {
          scriptOutput.textContent += '[Process aborted]';
        });
      };
                     // alert(`Saved to ${out1.value}, ${out2.value}, ${out3.value}`);
        scriptOutput.textContent='';
        const response=await fetch('/run-script',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
        list1_path:out1.value,
        list2_path:out2.value,
        list3_path:out3.value,
        bot1_path:bot1.value,
        bot2_path:bot2.value,
        bot3_path:bot3.value,
        segment_size:parseInt(segmentSizeInput.value),PSCS_file:PSCSFileInput.value,device_name:deviceNameInput.value,extra_arguments:extraArgumentsInput.value}),
                  signal: abortController.signal});
        //const reader=response.body.getReader();const decoder=new TextDecoder();
        //async function read(){const{done,value}=await reader.read();if(done) return;scriptOutput.textContent+=decoder.decode(value);read();}
        //read();
        
                const reader = response.body.getReader();
        const dec = new TextDecoder();
        async function read() {
          try {
            const {done, value} = await reader.read();
            if (done) return;
            scriptOutput.textContent += dec.decode(value);
            read();
          } catch(err) {
            console.log('Stream aborted');
          }
        }
        read();
      };
      
      
      

      
      document.getElementById('show-cmd').onclick = ()=>{ 
      
      fetch('/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            list1, list2, list3,
            bottom1, bottom2, bottom3,
            out1: out1.value, out2: out2.value, out3: out3.value,
            b1: bot1.value, b2: bot2.value, b3: bot3.value
          })
        });
      
      let cmdarr = ['python3','compute_concurrence.py', out1.value, out2.value, bot1.value, bot2.value, '--xy_pair_ids_flist_traval='+out3.value, '--xy_pair_ids_flist_tes='+bot3.value, `--w=${segmentSizeInput.value}`, `--device=${deviceNameInput.value}`, extraArgumentsInput.value];
      let val=PSCSFileInput.value;
      if (val.trim() !== ""){
        cmdarr.push(`--PSCS_file=${val}`)
      }
      cmd = cmdarr.join(' '); 
      scriptOutput.textContent = cmd; 
      };
      
      
      
      



      document.getElementById('copy-cmd').onclick=()=>{navigator.clipboard.writeText(scriptOutput.textContent).then(()=>alert('Copied'),()=>alert('Copy failed'));};


      function longestCommonSubstr(s1,s2){const m=s1.length,n=s2.length;let max=0,end=0;const tbl=Array(m+1).fill().map(()=>Array(n+1).fill(0));for(let i=1;i<=m;i++){for(let j=1;j<=n;j++){if(s1[i-1]===s2[j-1]){tbl[i][j]=tbl[i-1][j-1]+1;if(tbl[i][j]>max){max=tbl[i][j];end=i;}}}}return s1.slice(end-max,end);}  

      
    </script>
  </body>
</html>
"""

@app.route('/scan', methods=['POST'])
def scan():
    data = request.get_json()
    root = data.get('path', '')
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            files.append(os.path.join(dp, fn))
    files.sort()
    return jsonify(files)

@app.route('/split-bottom', methods=['POST'])
def split_bottom():
    data = request.get_json()
    for lst, key in [(data.get('bottom1', []), 'bot1'), (data.get('bottom2', []), 'bot2'), (data.get('bottom3', []), 'bot3')]:
        fname = data.get(key, f"{key}.txt")
        with open(fname, 'w') as f:
            for p in lst:
                f.write(p + '\n')
    return jsonify({'message': 'Wrote bottom lists to files'})

@app.route('/run-script', methods=['POST'])
def run_script():
    data = request.get_json()
    def generate():
        cmd = ['python3', 'compute_concurrence.py', data['list1_path'], data['list2_path'], 
        data['bot1_path'], data['bot2_path'], 
        f"--xy_pair_ids_flist_traval={data['list3_path']}",f"--xy_pair_ids_flist_tes={data['bot3_path']}", 
        f"--w={data.get('segment_size',0)}", f"--device={data.get('device_name',0)}"]

        if len(data['extra_arguments'])>0:
            cmd += data['extra_arguments'].split(' ')

        PSCS_file = data.get('PSCS_file')
        if len(PSCS_file) > 0:
            cmd.append('--PSCS_file')
            cmd.append(PSCS_file)
            
        global current_process
        with process_lock:
            current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        
        #process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in current_process.stdout:
            yield line
            
        with process_lock:
            current_process = None
            
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/abort', methods=['POST'])
def abort_process():
    global current_process
    with process_lock:
        if current_process and current_process.poll() is None:
            current_process.terminate()
            return jsonify({'status':'terminated'})
    return jsonify({'status':'no process'})

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        d = request.get_json()
        for arr, key in [(d.get('list1', []), 'out1'), (d.get('list2', []), 'out2'), (d.get('list3', []), 'out3'),
                         (d.get('bottom1', []), 'b1'), (d.get('bottom2', []), 'b2'), (d.get('bottom3', []), 'b3')]:
            fname = d.get(key)
            if fname:
                with open(fname,'w') as f:
                    for p in arr:
                        f.write(p + '\n')
        return 'OK'
    return render_template_string(HTML)

if __name__=='__main__':
    app.run(port=args.port)


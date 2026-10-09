const http=require('node:http');
const fs=require('node:fs');
const path=require('node:path');
function start({port=8080}={}){
    const base=path.resolve(__dirname,'..');
    const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8',
        '.json':'application/json; charset=utf-8','.png':'image/png','.svg':'image/svg+xml','.md':'text/plain; charset=utf-8'};
    const server=http.createServer((req,res)=>{
        if(req.method!=='GET'&&req.method!=='HEAD'){res.writeHead(405);res.end();return;}
        let file;
        try{const url=new URL(req.url,'http://localhost');const rel=decodeURIComponent(url.pathname);
            if(rel.includes('\\')||rel.split('/').some(p=>p.startsWith('.'))||rel.includes('/data/')||rel.includes('/node_modules/'))throw Error();
            file=path.resolve(base,'.'+(rel==='/'?'/index.html':rel));
            if(!file.startsWith(base+path.sep)||!mime[path.extname(file)])throw Error();
        }catch{res.writeHead(403);res.end('Forbidden');return;}
        fs.stat(file,(err,stat)=>{if(err||!stat.isFile()){res.writeHead(404);res.end('Not found');return;}
            res.writeHead(200,{'Content-Type':mime[path.extname(file)],'Cache-Control':'no-store'});
            if(req.method==='HEAD')res.end();else fs.createReadStream(file).pipe(res);
        });
    });
    server.on('error',error=>{console.error(error.code==='EADDRINUSE'?`Port ${port} is in use. Choose --port 8086 or another free port.`:error.message);process.exitCode=1;});
    server.listen(Number(port),'127.0.0.1',()=>console.log(`Brass + World Model: http://127.0.0.1:${port}`));
    return server;
}
module.exports={start};

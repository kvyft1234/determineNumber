
var mkRange = function(a){let sum=0; return Array(a).fill(1).map(function(k){return sum++;})};
Math.rand = function(){let r=Math.random(); return Math.log(r/(1-r))};
JSON.new = function(a){return JSON.parse(JSON.stringify(a));};
Array.sum = function(a){let sum=0; Array.num(a.length).map(function(k){sum += a[k]}); return sum;};
var mkTensor = function(a,...b){return b.length==0 ? Array(a).fill(0):Array(a).fill(0).map(i=>mkTensor(...b))};
var mkArray2 = function(arr){let r=mkTensor(arr.length); for(let i=0; i<arr.length; i++){r[i]=mkTensor(arr[i])} return r;};
var fillRandom = function(arr){return arr.map(i => Math.random()).map(i => Math.log(i/(1-i)))};
var cl = function (a, ...b){let c = Array.from(arguments).map(i=>JSON.new(i)); console.log.apply(this,c); return c[0];};

const AI = {};

AI.activationFunction = {};
AI.ANN = {};

AI.activationFunction.leakyReLU = function (x){return x>0 ? x:x*0.1};
AI.activationFunction.sigmoid = function (x){return 1/(1+Math.E**(-x))};
AI.activationFunction.tanh = Math.tanh;
AI.activationFunction.ReLU = x => x>0 ? x:0;

AI.ANN.variable = {};
AI.ANN.variable.layer = [];
AI.ANN.variable.perceptron = [];
AI.ANN.variable.weight = [];
AI.ANN.variable.bias = [];
AI.ANN.variable.z = [];
AI.ANN.variable.activationFunction = [];
AI.ANN.setLayer = function (layer){
	this.variable.layer = layer;
	let l = layer;
	let ll = l.length;
	let v = this.variable;
	let p = v.perceptron;
	let w = v.weight;
	let b = v.bias;
	let z = v.z;
	let a = v.activationFunction;
	let af = AI.activationFunction;
	let lr = af.tanh;
	let sm = af.sigmoid;
	p = mkArray2(l);
	w = mkRange(ll-1).map(i => mkTensor(l[i+1]).map(j => fillRandom(mkTensor(l[i]))));
	b = mkRange(ll-1).map(i => fillRandom(mkTensor(l[i+1])));
	z = mkRange(ll-1).map(i => mkTensor(l[i+1]));
	a = mkTensor(ll-1).map(i=>lr);
	a[ll-2] = sm;
	this.variable.perceptron = p;
	this.variable.weight = w;
	this.variable.bias = b;
	this.variable.z = z;
	this.variable.activationFunction = a;
};
AI.ANN.propagation = function(inputLayer){
	let v = this.variable;
	let l = v.layer;
	let ll = l.length;
	let p = v.perceptron;
	let w = v.weight;
	let b = v.bias;
	let z = v.z;
	let a = v.activationFunction;
	p[0] = inputLayer;
	for(let i = 0; i < ll-1; i++){
		for(let j = 0; j < l[i+1]; j++){
			let sum = 0;
			for(let k = 0; k < l[i]; k++){
				sum += p[i][k] * w[i][j][k];
			}
			sum += b[i][j];
			z[i][j] = sum;
			p[i+1][j] = a[i](z[i][j]);
		}
	}
	this.variable.perceptron = p;
	this.variable.z = z;
	return p[ll-1];
};
AI.ANN.backPropagation = function (answer, learnRate){
	let v = this.variable;
	let l = v.layer;
	let ll = l.length;
	let p = v.perceptron;
	let w = v.weight;
	let b = v.bias;
	let z = v.z;
	let a = v.activationFunction;
	let delta = mkRange(ll-1).map(i => mkTensor(l[i+1]));
	let df = function(f,x){return (f(x+0.001)-f(x-0.001))/0.002};
	let daf = function(n,x){return df(a[n],x)};
	for(let i=0; i<l[ll-1]; i++){
		delta[ll-2][i] = 2*(p[ll-1][i]-answer[i])*daf(ll-2,z[ll-2][i]);
	}
	for(let i=0; i<l[ll-1]; i++){
		for(let j=0; j<l[ll-2]; j++){
			w[ll-2][i][j] -= learnRate * delta[ll-2][i] * p[ll-2][j];
		}
		b[ll-2][i] -= learnRate * delta[ll-2][i];
	}
	for(let i=ll-3; i>=0; i--){
		for(let j=0; j<l[i+1]; j++){
			for(let k=0; k<l[i+2]; k++){
				delta[i][j] += delta[i+1][k]*w[i+1][k][j];
			}
			delta[i][j] *= daf(i,z[i][j]);
		}
		for(let j=0; j<l[i+1]; j++){
			for(let k=0; k<l[i]; k++){
				w[i][j][k] -= learnRate * delta[i][j] * p[i][k];
			}
			b[i][j] -= learnRate * delta[i][j];
		}
	}
	this.variable.weight = w;
	this.variable.bias = b;
	let cost = 0;
	for(let i=0; i<p[ll-1].length; i++){
		cost += (p[ll-1][i]-answer[i])**2;
	}
};

AI.ANN.setLayer([2,8,1]);
// AI.ANN.variable.weight=[[[10,10],[-10,10],[-10,-10],[10,-10]],[[-10,10,-10,10]]];
// AI.ANN.variable.bias=[[0,0,0,0],[0]];

var time;
function learn(){
	clearInterval(time);
	time = setInterval(function(){
	for(let _$=0; _$<100; _$++){
		for(let n=0; n<1000; n++){
			AI.ANN.propagation([0,0]);
			AI.ANN.backPropagation([0], 0.0008);
			AI.ANN.propagation([0.4,0]);
			AI.ANN.backPropagation([0], 0.0008);
			AI.ANN.propagation([0.6,0]);
			AI.ANN.backPropagation([1], 0.0008);
			AI.ANN.propagation([1,0]);
			AI.ANN.backPropagation([1], 0.0008);

			AI.ANN.propagation([0,0.4]);
			AI.ANN.backPropagation([0], 0.0008);
			AI.ANN.propagation([0.4,0.4]);
			AI.ANN.backPropagation([0], 0.0008);
			AI.ANN.propagation([0.6,0.4]);
			AI.ANN.backPropagation([1], 0.0008);
			AI.ANN.propagation([1,0.4]);
			AI.ANN.backPropagation([1], 0.0008);

			AI.ANN.propagation([0,0.6]);
			AI.ANN.backPropagation([1], 0.0008);
			AI.ANN.propagation([0.4,0.6]);
			AI.ANN.backPropagation([1], 0.0008);
			AI.ANN.propagation([0.6,0.6]);
			AI.ANN.backPropagation([0], 0.0008);
			AI.ANN.propagation([1,0.6]);
			AI.ANN.backPropagation([0], 0.0008);

			AI.ANN.propagation([0,1]);
			AI.ANN.backPropagation([1], 0.0008);
			AI.ANN.propagation([0.4,1]);
			AI.ANN.backPropagation([1], 0.0008);
			AI.ANN.propagation([0.6,1]);
			AI.ANN.backPropagation([0], 0.0008);
			AI.ANN.propagation([1,1]);
			AI.ANN.backPropagation([0], 0.0008);

		}
	}
		cl((AI.ANN.propagation([1,0])[0]-1)**2+(AI.ANN.propagation([0,1])[0]-1)**2+AI.ANN.propagation([0,0])[0]**2+AI.ANN.propagation([1,1])[0]**2);
	},3000);
	setTimeout(function(){
		clearInterval(time);
		cl('complete')
	},20000);
}

var a = document.createElement('canvas');
a.setAttribute('width',"100px");
a.setAttribute('height',"100px");
a.setAttribute('style',"border: 1px solid black;");
var b = document.querySelector("body > div.Network");
b.appendChild(a);
var c = document.querySelector('canvas');
var d = c.getContext('2d');

function setColor(color){d.fillStyle = `rgb(${Math.floor(color*255)},255,${Math.floor(color*255)})`};

function view(){
	d.clearRect(0,0,100,100);
	for(let i=0; i<101; i++){ for(let j=101; j>=0; j--){
		setColor(1-AI.ANN.propagation([i/100,j/100])[0]);
		d.beginPath();
		d.fillRect(i,j,1,1);
		d.closePath();
	}}
}
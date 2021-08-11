var select = function(a,...b){let n=document.querySelectorAll(a); return b[0]==undefined ? n[0]:n[b[0]]};
var cl = function(a,...b){console.log(a,...b); return a};

var cvs = select("#draw");
var ctx = cvs.getContext('2d');
ctx.ondraw = false;
ctx.gap = 28/400;

cvs.ondblclick = function(e){
	ctx.clearRect(0,0,28,28);
};
cvs.onmousedown = function(e){
	//ctx.clearRect(0,0,28,28);
	ctx.beginPath();
	ctx.moveTo(e.layerX*ctx.gap, e.layerY*ctx.gap);
	ctx.ondraw = true;
};
cvs.onmousemove = function(e){
	if(ctx.ondraw){
		ctx.lineTo(e.layerX*ctx.gap, e.layerY*ctx.gap);
		ctx.stroke();
	}
};
cvs.onmouseup = function(e){
	ctx.lineTo(e.layerX*ctx.gap, e.layerY*ctx.gap);
	ctx.stroke();
	ctx.closePath();
	ctx.ondraw = false;
}

var mkRange = function(a){let sum=0; return Array(a).fill(1).map(function(k){return sum++;})};
Math.rand = function(){let r=Math.random(); return Math.log(r/(1-r))};
JSON.new = function(a){return JSON.parse(JSON.stringify(a));};
Array.sum = function(a){let sum=0; Array.num(a.length).map(function(k){sum += a[k]}); return sum;};
var mkTensor = function(a,...b){return b.length==0 ? Array(a).fill(0):Array(a).fill(0).map(i=>mkTensor(...b))};
var mkArray2 = function(arr){let r=mkTensor(arr.length); for(let i=0; i<arr.length; i++){r[i]=mkTensor(arr[i])} return r;};
var fillRandom = function(arr){
	return arr.map(i => Math.random()).map(i => Math.log(i/(1-i))/2);
};

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
AI.ANN.propagation = function(iL){
	let v = this.variable;
	let l = v.layer;
	let ll = l.length;
	let p = v.perceptron;
	let w = v.weight;
	let b = v.bias;
	let z = v.z;
	let a = v.activationFunction;
	p[0] = iL;
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

function extractImage(){
	let k = ctx.getImageData(0,0,28,28).data;
	let r = mkTensor(28,28);
	for(let i=0; i<28; i++){
		for(let j=0; j<28; j++){
			r[i][j]=k[(i*28+j)*4+3];
		}
	}
	return r;
};
function arrayFlipX(arr){
	let y=arr.length, x=arr[0].length;
	let r = mkTensor(y,x);
	for(let i=0; i<y; i++){
		for(let j=0; j<x; j++){
			r[i][j]=arr[i][x-j-1];
		}
	}
	return r;
};
function arrayFlipY(arr){
	let y=arr.length, x=arr[0].length;
	let r = mkTensor(y,x);
	for(let i=0; i<y; i++){
		for(let j=0; j<x; j++){
			r[i][j]=arr[y-i-1][j];
		}
	}
	return r;
};
function transposedArray(arr){
	let y=arr.length, x=arr[0].length;
	let r = mkTensor(y,x);
	for(let i=0; i<y; i++){
		for(let j=0; j<x; j++){
			r[i][j]=arr[j][i];
		}
	}
	return r;
};
var testSet;
var filter;
filter = [[[1,1,1,1],[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1]],[[1,1,1,-1],[1,1,-1,-1],[1,1,-1,-1],[1,-1,-1,-1]]];
filter.push(arrayFlipY(filter[0]));
filter.push(transposedArray(filter[0]));
filter.push(transposedArray(filter[2]));

filter.push(transposedArray(filter[1]));
filter.push(arrayFlipX(filter[5]));
filter.push(arrayFlipX(filter[1]));
filter.push(arrayFlipY(filter[1]));
filter.push(arrayFlipY(filter[5]));
filter.push(arrayFlipY(filter[6]));
filter.push(arrayFlipY(filter[7]));

function convolution(image,filter){
	let iy = image.length, ix = image[0].length, fy = filter.length, fx = filter[0].length;
	let y = iy-fy+1, x = ix-fx+1;
	let r = mkTensor(y,x);
	let sum;
	for(let i = 0; i < y; i++){
		for(let j = 0; j < x; j++){
			sum = 0;
			for(let k = 0; k < fy; k++){
				for(let l = 0; l < fx; l++){
					sum+=image[i+k][j+l]*filter[k][l];
				}
			}
			r[i][j]=sum;
		}
	}
	return r;
};
function maxPooling(convolutedImage){
	let y = Math.floor(convolutedImage.length/2), x = Math.floor(convolutedImage[0].length/2);
	let r = mkTensor(y,x);
	let max, temp;
	for(let i=0; i<y; i++){
		for(let j=0; j<x; j++){
			max=0;
			for(let k=0; k<2; k++){
				for(let l=0; l<2; l++){
					temp=convolutedImage[i*2+k][j*2+l];
					if(temp>max)max=temp;
				}
			}
			r[i][j]=max>0?max/50:0;
		}
	}
	return r;
}
var convolutioned;
function step1(){
	testSet = extractImage();
	filter = [[[1,1,1,1],[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1]],[[1,1,1,-1],[1,1,-1,-1],[1,1,-1,-1],[1,-1,-1,-1]]];
	filter.push(arrayFlipY(filter[0]));
	filter.push(transposedArray(filter[0]));
	filter.push(transposedArray(filter[2]));

	filter.push(transposedArray(filter[1]));
	filter.push(arrayFlipX(filter[5]));
	filter.push(arrayFlipX(filter[1]));
	filter.push(arrayFlipY(filter[1]));
	filter.push(arrayFlipY(filter[5]));
	filter.push(arrayFlipY(filter[6]));
	filter.push(arrayFlipY(filter[7]));
	convolutioned = [[],[]];
	for(let i=0; i<filter.length; i++){
		convolutioned[0].push(maxPooling(convolution(testSet,filter[i])));
	}
}

var inputLayer;
function step2(){
	filter = [[[1,-1,-1,1],[1,-1,-1,1],[1,-1,-1,1],[1,-1,-1,1]],[[-1,-1,-1,1],[1,-1,-1,1],[1,-1,-1,1],[1,-1,-1,-1]]];
	filter.push(arrayFlipY(filter[0]));
	filter.push(transposedArray(filter[0]));
	filter.push(transposedArray(filter[2]));
	filter.push(transposedArray(filter[1]));
	filter.push(arrayFlipX(filter[5]));
	filter.push(arrayFlipX(filter[1]));
	filter.push(arrayFlipY(filter[1]));
	filter.push(arrayFlipY(filter[5]));
	filter.push(arrayFlipY(filter[6]));
	filter.push(arrayFlipY(filter[7]));
	filter.push([[1,1,1,1],[1,-1,-1,1],[1,-1,-1,1],[1,1,1,1]]);
	for(let i=0; i<convolutioned[0].length; i++){
		for(let j=0; j<filter.length; j++){
			convolutioned[1].push(maxPooling(convolution(convolutioned[0][i],filter[j])).flat());
		}
	}
	convolutioned.push([]);
	convolutioned[2] = mkTensor(convolutioned[1][0].length);
	for(let i = 0; i < convolutioned[1].length; i++){
		for(let j = 0; j < convolutioned[1][i].length; j++){
			convolutioned[2][j] += (convolutioned[1][i][j]-0.6)/20;
		}
	}
	inputLayer = JSON.new(convolutioned[2]);
}

function mkResultArray(n){
	let r = Array(10).fill(0);
	r[n] = 1;
	return r;
}

AI.ANN.setLayer([16,50,10]);

// function step3(n){
// 	step1();
// 	step2();
// 	AI.ANN.propagation(inputLayer);
// 	for(let i=0; i<1000; i++){
// 		AI.ANN.backPropagation(mkResultArray(n), 0.08);
// 	}
// }

var convolutionedSet = mkTensor(10,0);

function showExpectation(){
	let t = AI.ANN.propagation(inputLayer);
	let index=0, max = 0;
	for(let i=0; i<10; i++){
		if(t[i] > max){
			max = t[i];
			index = i;
		}
	}
	select("#expection").innerText = index + " : " + JSON.stringify(t);
}

document.onkeydown = function(e){
	if(!e.shiftKey){
		if(!Number.isNaN(Number(e.key))){
			step1();
			step2();
			convolutionedSet[Number(e.key)].push(inputLayer);
		}
	}else{
		step1();
		step2();
		cl("expecting...");
		showExpectation();
	}
};

var learnMacro = function(){
	console.log("executing...");
	for(let n=0; n<100; n++){
		console.log(n+"%")
		for(let i=0; i<10; i++){
			for(let j=0; j<convolutionedSet[i].length; j++){
				for(let k=0; k<20; k++){
					AI.ANN.propagation(convolutionedSet[i][j]);
					AI.ANN.backPropagation(mkResultArray(i), 1);
				}
			}
		}
	}
	console.log("done!");
}

function similarity(arr,n){
	let sum = 0;
	for(let i=0; i<10; i++){
		sum += (n==i?1:-1)*arr[i];
	}
	return sum;
}

// convolutionedSet = data;
// learnMacro();





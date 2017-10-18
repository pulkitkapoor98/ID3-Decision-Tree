import weka.core.*;
import weka.core.converters.ArffLoader.ArffReader;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

class Node{
	String value;
	int index;
	boolean is_leaf;
	String operator;
	String type;
	String operand;
	int lNeg;
	int uPos;
	ArrayList<Node> candidateSplit;
	Node(){
		this.value = null;
		this.index = -1;
		this.is_leaf = false;
		this.operator = null;
		this.type = null;
		this.operand = null;
		this.lNeg = 0;
		this.uPos = 0;
		this.candidateSplit = new ArrayList<Node>();
	}
	Node(String val,int index,Boolean is_leaf,String operator,String type,String operand){
		this.value = val;
		this.index = index;
		this.is_leaf = is_leaf;
		this.operator = operator;
		this.type = type;
		this.operand = operand;
		this.lNeg = 0;
		this.uPos = 0;
		this.candidateSplit = new ArrayList<Node>();
	}
}

class GeneratePlot {
	Instances trainData;
	Instances testData;
	public GeneratePlot(Instances trainData,Instances testData){
		this.trainData = new Instances(trainData);
		this.testData = new Instances(testData);
	}
	public DecisionTree generateTreeofSample(Instances sample,int m){
		DecisionTree dTrainTree = new DecisionTree();
		dTrainTree.data = sample;
		HashSet<Attribute> attrSet = new HashSet<Attribute>(); 
		for(int i=0;i<dTrainTree.data.numAttributes()-1;i++){
			attrSet.add(dTrainTree.data.attribute(i));
		}
		dTrainTree.Root = dTrainTree.generateDecisionTree(dTrainTree.data,dTrainTree.data.classAttribute(), attrSet,m,null);
		return dTrainTree;
	}
	public void generatePlotsPart2(){
		try (BufferedWriter bw = new BufferedWriter(new FileWriter("ques2.txt"))) {
			String content = "x\ty\n";
			bw.write(content);
			int[] xaxis = {5,10,20,50};
			int sample_size;
			int count = 1;
			Instances sample;
			DecisionTree dTrainTree;
			Random rand = new Random();
			ArrayList<String> actual;
			ArrayList<String> predicted;
			int totalPred;
			int match=0;
			double sum;
			ArrayList<Double> accuracy;
			for(int i=0;i<xaxis.length;i++){
				sample_size = (xaxis[i]* trainData.size())/100;
				count = 10;
				accuracy = new ArrayList<Double>();
				while(count > 0){
					int random = rand.nextInt(trainData.size()-sample_size);
					sample = new Instances(trainData,random,sample_size);
					dTrainTree = generateTreeofSample(sample,4);
					actual =  new ArrayList<String>();
					predicted =  new ArrayList<String>();
					for(int j=0;j<this.testData.size();j++){
						actual.add(this.testData.get(j).stringValue(this.testData.classIndex()));
						predicted.add(dTrainTree.predict(dTrainTree.Root,this.testData.get(j)));
					}
					totalPred = predicted.size();
					match = 0;
					for(int j=0;j<totalPred;j++){
						if(actual.get(j).equals(predicted.get(j))){
							match++;
						}
					}
					accuracy.add((double)match/totalPred);
					count--;
				}
				Collections.sort(accuracy);
				sum = 0;
				for(Double d : accuracy)
				    sum += d;
				bw.write(xaxis[i]+"\t"+accuracy.get(0)+"\n");
				bw.write(xaxis[i]+"\t"+(sum/accuracy.size())+"\n");
				bw.write(xaxis[i]+"\t"+accuracy.get(9)+"\n");
			}
			dTrainTree = generateTreeofSample(trainData,4);
			actual =  new ArrayList<String>();
			predicted =  new ArrayList<String>();
			for(int j=0;j<this.testData.size();j++){
				actual.add(this.testData.get(j).stringValue(this.testData.classIndex()));
				predicted.add(dTrainTree.predict(dTrainTree.Root,this.testData.get(j)));
			}
			totalPred = predicted.size();
			match = 0;
			for(int j=0;j<totalPred;j++){
				if(actual.get(j).equals(predicted.get(j))){
					match++;
				}
			}
			bw.write(100+"\t"+((double)match/totalPred)+"\n");
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
	}
	public void generatePlotsPart3(){
		try (BufferedWriter bw = new BufferedWriter(new FileWriter("ques3.txt"))) {
			String content = "x\ty\n";
			bw.write(content);
			int[] xaxis = {2,5,10,20};
			DecisionTree dTrainTree;
			ArrayList<String> actual;
			ArrayList<String> predicted;
			int totalPred;
			int match=0;
			for(int i=0;i<xaxis.length;i++){
				dTrainTree = generateTreeofSample(this.trainData,xaxis[i]);
				actual =  new ArrayList<String>();
				predicted =  new ArrayList<String>();
				for(int j=0;j<this.testData.size();j++){
					actual.add(this.testData.get(j).stringValue(this.testData.classIndex()));
					predicted.add(dTrainTree.predict(dTrainTree.Root,this.testData.get(j)));
				}
				totalPred = predicted.size();
				match = 0;
				for(int j=0;j<totalPred;j++){
					if(actual.get(j).equals(predicted.get(j))){
						match++;
					}
				}
				bw.write(xaxis[i]+"\t"+((double)match/totalPred)+"\n");
			}
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
	}
}
public class DecisionTree {
	Instances data;
	ArrayList<Node> Root;
	public DecisionTree(){
		Root = new ArrayList<Node>();
	}
	public void readFile(String FilePath){
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(FilePath));
			ArffReader arff = new ArffReader(br);
			data = arff.getData();		
 			data.setClassIndex(data.numAttributes() - 1);
		}
		catch (IOException e) {
			e.printStackTrace();
		} 
	}
	public ArrayList<Instance> convertToList(Instances data){
		int i;
		ArrayList<Instance> dataList = new ArrayList<Instance>();
		for(i=0;i<data.size();i++){
			dataList.add(data.get(i));
		}
		return dataList;
	}
	public double entropy(int neg,int pos){
		if(pos==0 || neg==0)
			return 0.0;
		double ent = 0.0;
		double total = pos+neg;
		ent = (-(pos/total)*(Math.log((double)pos/total)/Math.log(2))) - ((neg/total)*(Math.log((double)neg/total)/Math.log(2)));
		if(Double.isNaN(ent))
			return 0.0;
		else
			return ent;
	}
	public HashSet<String> checkNumericValue(double[] arr,double val,Instances d,int attrIndex){
		HashSet<String> h = new HashSet<String>();
		for(int i=0;i<arr.length;i++){
			if(arr[i]==val){
				h.add(d.get(i).stringValue(d.classIndex()));
			}
		}
		return h;
	}
	public ArrayList<Double> entropyCandidateSplits(Instances d,Attribute attr,ArrayList<Double> candidateSplits){
		ArrayList<Double> entropyCandidateSplits = new ArrayList<Double>();
		int gPos = 0;
		int gNeg = 0;
		int sPos = 0;
		int sNeg = 0;
		double small,large;
		for(Double num : candidateSplits){
			gPos = 0;
			gNeg = 0;
			sPos = 0;
			sNeg = 0;
			for(int i=0;i<d.size();i++){
				if(d.get(i).value(attr.index()) <= num){
					if(d.get(i).stringValue(d.get(i).classIndex()).equals(d.classAttribute().value(0))){
						gNeg++;
					}
					else{
						gPos++;
					}
				}
				else{
					if(d.get(i).stringValue(d.get(i).classIndex()).equals(d.classAttribute().value(0))){
						sNeg++;
					}
					else{
						sPos++;
					}
				}
			}
			small=((double)(sPos+sNeg)/(double)(d.size())) * entropy(sNeg, sPos);
			large=((double)(gPos+gNeg)/(double)(d.size())) * entropy(gNeg, gPos);
			entropyCandidateSplits.add(small+large);
		}
		return entropyCandidateSplits;
	}
	public ArrayList<ArrayList<Double>> infoNumericGain(Attribute attr,Instances d){
		double[] attrArray = d.attributeToDoubleArray(attr.index());
		SortedSet<Double> set = new TreeSet<Double>();
		for(int i=0;i<attrArray.length;i++){
			set.add(attrArray[i]);
		}
		HashSet<String> hLower;
		HashSet<String> hUpper;
		double lower = 0.0;
		double upper = 0.0;
		ArrayList<Double> list = new ArrayList<Double>();
		lower=set.first();
		set.remove(lower);
		Iterator it = set.iterator();
		while (it.hasNext()) {
			hLower = checkNumericValue(attrArray,lower,d,attr.index());
			upper = (double)it.next();
			hUpper = checkNumericValue(attrArray,upper,d,attr.index());
			hLower.addAll(hUpper);
			if(hLower.size() > 1){
				list.add((upper+lower)/2);
			}
			lower = upper;
		}
		ArrayList<Double> entropyCandidateSplits = entropyCandidateSplits(d,attr,list);
		if(entropyCandidateSplits.isEmpty())
			return null;
		ArrayList<ArrayList<Double>> arr = new ArrayList<ArrayList<Double>>();
		arr.add(entropyCandidateSplits);
		arr.add(list);
		return arr;
	}
	public double infoNominalGain(Attribute attr,Instances d,double totalEnt){
		int i;
		String attrValue;
		Map<String,ArrayList<Instance>> m = new HashMap<String,ArrayList<Instance>>();
		ArrayList<Instance> nestedArr;
		for(i=0;i<attr.numValues();i++){
			nestedArr = new ArrayList<Instance>();
			m.put(attr.value(i), nestedArr);
		}
		
		for(i=0;i<d.size();i++){
			attrValue = d.get(i).stringValue(attr.index());
			for(Map.Entry<String,ArrayList<Instance>> hm:m.entrySet()){ 
				if(hm.getKey().equals(attrValue)){
					hm.getValue().add(d.get(i));
				}
			}
		}
		int neg,pos;
		i=0;
		double infoGain=0.0;
		double total = 0.0;
		for(Map.Entry<String,ArrayList<Instance>> hm:m.entrySet()){
			neg = countNeg(hm.getValue());
			pos = hm.getValue().size() - neg;
			total = total + (((double)(neg+pos)/d.size()) * entropy(neg,pos));
		}
		return total;
	}
	public int countNeg(ArrayList<Instance> arr){
		int cnt=0;
		for(Instance i : arr){
			if(i.stringValue(i.classIndex()).equals(data.classAttribute().value(0))){
				cnt++;
			}
		}
		return cnt;
	}
	public int getMinIndex(ArrayList<Double> arr){
		int minIndex=0;
		double min=arr.get(0);
		for(int i=1;i<arr.size();i++){
			if(arr.get(i)<min){
				min = arr.get(i);
				minIndex = i;
			}
		}
		return minIndex;
	}
	public Attribute calInfoGain(Instances data){
		ArrayList<Instance> totalList = convertToList(data);
		double totalEnt = entropy(countNeg(totalList),totalList.size()-countNeg(totalList));
		double minEntropy=Double.MAX_VALUE;
		double entropy=0.0;
		Attribute splitAtr = null;
		for(int i=0;i<data.numAttributes()-1;i++){
				if(data.attribute(i).isNumeric()){
					ArrayList<ArrayList<Double>> arr = infoNumericGain(data.attribute(i),data);
					if(arr == null)
						entropy = 1.0;
					else{
						int minIndex = getMinIndex(arr.get(0));
						entropy = arr.get(0).get(minIndex);
					}
				}
				else if(data.attribute(i).isNominal()){
					entropy = infoNominalGain(data.attribute(i),data,totalEnt);
				}
				if(entropy<minEntropy){
					minEntropy = entropy;
					splitAtr = data.attribute(i);
				}
			
		}
		return splitAtr;
	}
	public boolean isPositive(Instances d,Attribute target){
		for(int i=0;i<d.size();i++){
			if(d.get(i).stringValue(d.get(i).classIndex()).equals(target.value(0))){
				return false;
			}
		}
		return true;
	}
	public boolean isNegative(Instances d,Attribute target){
		for(int i=0;i<d.size();i++){
			if(d.get(i).stringValue(d.get(i).classIndex()).equals(target.value(1))){
				return false;
			}
		}
		return true;
	}
	public String mostCommon(Instances d,Attribute target){
		int pos=0;
		int neg=0;
		for(int i=0;i<d.size();i++){
			if(d.get(i).stringValue(d.get(i).classIndex()).equals(target.value(1))){
				pos++;
			}
			else
				neg++;
		}
		if(pos==neg)
			return null;
		else
			return (pos<neg?target.value(0):target.value(1));
	}
	public double getThreshold(Instances d,Attribute atr){
		ArrayList<ArrayList<Double>> arr = infoNumericGain(atr,d);
		if(arr == null){
			return -1;
		}
		int minIndex = getMinIndex(arr.get(0));
		return arr.get(1).get(minIndex);
	}
	public Instances findSubset(Instances d,Attribute atr,String val){
		double threshold = 0.0;
		Instances resultdt = new Instances(d);
		if(atr.isNumeric()){
			ArrayList<ArrayList<Double>> arr = infoNumericGain(atr,d);
			if(arr == null)
				return null;
			int minIndex = getMinIndex(arr.get(0));
			threshold = arr.get(1).get(minIndex);
		}
		for(int i=resultdt.size()-1;i>=0;i--){
			if(atr.isNominal()){
				if(!resultdt.get(i).stringValue(atr.index()).equals(val)){
					resultdt.delete(i);
				}
			}
			if(atr.isNumeric()){
				if(val.equals("less")){
					if(resultdt.get(i).value(atr.index()) > threshold){
						resultdt.delete(i);
					}
				}
				if(val.equals("more")){
					if(resultdt.get(i).value(atr.index()) <= threshold){
						resultdt.delete(i);
					}
				}
			}
		}
		return resultdt;
	}
	public ArrayList<Node> generateDecisionTree(Instances d,Attribute target,HashSet<Attribute> attrSet,int m,String mostCommonClass){
		ArrayList<Node> rootList = new ArrayList<Node>();
		if(d.size()<m){
			String mostCom = mostCommon(d,target);
			if(mostCom == null){
				mostCom = mostCommonClass;
			}
			Node newNode = new Node(mostCom,-1,true,null,null,null);
			rootList.add(newNode);
			return rootList;
		}
		if(isPositive(d,target)){
			Node newNode = new Node("positive",-1,true,null,null,null);
			rootList.add(newNode);
			return rootList;
		}
		if(isNegative(d,target)){
			Node newNode = new Node("negative",-1,true,null,null,null);
			rootList.add(newNode);
			return rootList;
		}
		if(attrSet.isEmpty()){
			String mostCom = mostCommon(d,target);
			if(mostCom == null){
				mostCom = mostCommonClass;
			}
			Node newNode = new Node(mostCom,-1,true,null,null,null);
			rootList.add(newNode);
			return rootList;
		}
		Attribute split = calInfoGain(d);
		ArrayList<Instance> convertList = convertToList(d);
		String mostCommoninParent = mostCommonClass;
		int negCnt = countNeg(convertList);
		if(negCnt > d.size()-negCnt){
			mostCommoninParent = "negative";
		}
		else if(negCnt < d.size()-negCnt){
			mostCommoninParent = "positive";
		}
		if(split.isNominal()){
			for(int i=0;i<split.numValues();i++){
				Node newNode = new Node(split.name(),split.index(),false,"=","Nominal",split.value(i));
				Instances resultdt = findSubset(d,split,split.value(i));
				ArrayList<Instance> arr = convertToList(resultdt);
				newNode.lNeg = countNeg(arr);
				newNode.uPos = arr.size()-countNeg(arr);
				rootList.add(newNode);
				newNode.candidateSplit.addAll(generateDecisionTree(resultdt,target,attrSet,m,mostCommoninParent));
			}
		}
		if(split.isNumeric()){
			String val = "less";
			double threshold = getThreshold(d,split);
			if(threshold == -1){
				String mostCom = mostCommon(d,target);
				if(mostCom == null){
					mostCom = mostCommonClass;
				}
				Node newNode = new Node(mostCom,-1,true,null,null,null);
				rootList.add(newNode);
				return rootList;
			}
			for(int i=0;i<2;i++){
				Node newNode;
				if(val.equals("less")){
					newNode = new Node(split.name(),split.index(),false,"<=","Numeric",Double.toString(threshold));
				}
				else{
					newNode = new Node(split.name(),split.index(),false,">","Numeric",Double.toString(threshold));
				}
				Instances resultdt = findSubset(d,split,val);
				ArrayList<Instance> arr = convertToList(resultdt);
				newNode.lNeg = countNeg(arr);
				newNode.uPos = arr.size()-countNeg(arr);
				rootList.add(newNode);
				newNode.candidateSplit.addAll(generateDecisionTree(resultdt,target,attrSet,m,mostCommoninParent));
				val = "more";
			}
		}
		return rootList;
	}
	public void printTree(ArrayList<Node> root, int t,boolean flag){
		int tabs = t;
		for(int i=0;i<root.size();i++){
			if(!root.get(i).is_leaf){
				if(flag){
					System.out.println();
				}
			}
			flag=true; 
			t=tabs;
			while(t>0 && !root.get(i).is_leaf){
				System.out.printf("|\t");
				t--;
			}
			if(root.get(i).is_leaf)
				System.out.print(": "+root.get(i).value);
			else{
				if(root.get(i).type.equals("Nominal"))
					System.out.print(root.get(i).value+" "+root.get(i).operator+" "+root.get(i).operand+" ["+root.get(i).lNeg+" "+root.get(i).uPos+"]");
				else{
					System.out.print(root.get(i).value+" "+root.get(i).operator+" ");
					System.out.format("%.6f", Double.parseDouble((root.get(i).operand)));  
					System.out.print(" ["+root.get(i).lNeg+" "+root.get(i).uPos+"]");
				}
			}	
			printTree(root.get(i).candidateSplit,tabs+1,true);
		}
	}
	public String predict(ArrayList<Node> root,Instance d){
		for(int i=0;i<root.size();i++){
			if(root.get(i).is_leaf){
				return root.get(i).value;
			}
			if(root.get(i).type.equals("Nominal")){
				if(d.stringValue(root.get(i).index).equals(root.get(i).operand)){
					return (predict(root.get(i).candidateSplit,d));
				}
			}
			else{
				if(root.get(i).operator.equals("<=")){
					if(d.value(root.get(i).index) <= Double.parseDouble(root.get(i).operand)){
						return (predict(root.get(i).candidateSplit,d));
					}
				}
				else{
					if(d.value(root.get(i).index) > Double.parseDouble(root.get(i).operand)){
						return (predict(root.get(i).candidateSplit,d));
					}
				}
			}
		}
		return null;
	}
	public static void main(String[] args){
		if(args.length == 3){
			String trainFilePath = args[0];
			DecisionTree dTree = new DecisionTree();
			dTree.readFile(trainFilePath);
			HashSet<Attribute> attrSet = new HashSet<Attribute>(); 
			for(int i=0;i<dTree.data.numAttributes()-1;i++){
				attrSet.add(dTree.data.attribute(i));
			}
			int m = Integer.parseInt(args[2]);
			dTree.Root = dTree.generateDecisionTree(dTree.data,dTree.data.classAttribute(), attrSet,m,null);
			dTree.printTree(dTree.Root,0,false);
			System.out.println();
			System.out.println("<Predictions for the Test Set Instances>");
			String testFilePath = args[1];
			DecisionTree dTreeTest = new DecisionTree();
			dTreeTest.readFile(testFilePath);
			int cnt = 0;
			for(int i=0;i<dTreeTest.data.size();i++){
				if(dTreeTest.data.get(i).stringValue(dTreeTest.data.classIndex()).equals(dTreeTest.predict(dTree.Root,dTreeTest.data.get(i)))){
					cnt++;
				}
				System.out.println((i+1)+": Actual: "+dTreeTest.data.get(i).stringValue(dTreeTest.data.classIndex())+" Predicted: "+dTreeTest.predict(dTree.Root,dTreeTest.data.get(i)));
			}
			System.out.print("Number of correctly classified: "+cnt+" Total number of test instances: "+dTreeTest.data.size());
			//GeneratePlot plot = new GeneratePlot(dTree.data,dTreeTest.data);
			//plot.generatePlotsPart2();
			//plot.generatePlotsPart3();
		}
		else{
			System.out.println("Invalid count of arguments");
		}
	}
}
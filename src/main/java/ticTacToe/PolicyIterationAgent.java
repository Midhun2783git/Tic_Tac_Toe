package ticTacToe;


import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{
		/*
		 * YOUR CODE HERE
		 */
	    for (Game state : curPolicy.keySet()) {   // Iterate through all states
	        if (!state.isTerminal()) {    // if not a terminal state
	            List<Move> posMvs = state.getPossibleMoves();  // retrieve all moves from state 
	            if (!posMvs.isEmpty()) {  //if move not empty
	                int randomIndex = (int) (Math.random() * posMvs.size());  //Retrieve random move from index
	                Move randomMove = posMvs.get(randomIndex);
	                curPolicy.put(state, randomMove); // Assign the randomly chosen move to the state
	            }
	        }
	    }
	}
	
	private double calcVal(List<TransitionProb> tr) {
		double val = 0.0;
		
		for(TransitionProb t : tr) {
			val += t.prob * (t.outcome.localReward +(discount * policyValues.get(t.outcome.sPrime))); 
		}
		return val;
	}
	
	private double calcStVal(Game state) {
	    double stVal = 0.0;

	    Move poMove = curPolicy.get(state);
	    for (TransitionProb tr : mdp.generateTransitions(state, poMove)) {
	        double tVal = tr.prob * (tr.outcome.localReward + discount * policyValues.getOrDefault(tr.outcome.sPrime, 0.0));
	        stVal += tVal;
	    }

	    return stVal;
	}
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the currrent policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)
	{
		/* YOUR CODE HERE */
		boolean cVal = false; //initializing convergence value to false
		while (!cVal) { // evaluating until convergence 
		    double max = 0.0; //initializing maximum value 
		    for (Game st : curPolicy.keySet()) {  // iterating through states of current policy 
		        if (!st.isTerminal()) {   // if not a terminal state
		            double prevVal = policyValues.getOrDefault(st, 0.0);	//Retrieving previous value of state	   
		            double val = calcStVal(st);  //Calculating value of state
		            policyValues.put(st, val);	// Updating policy with state and values 	           
		            double eVal = Math.abs(val - prevVal); //calculating the evaluated value by subtracting the value from previous value
		            if (eVal > max) { //Updating the maximum value if evaluated value is greater 
		                max = eVal;
		            }
		        }
		    }
		    if (max <= delta) { //if maximum value is less than or equal to delta 
		        cVal = true;  	// updating convergence as true
		    }
		}
	}
		
		
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
		/* YOUR CODE HERE */
		boolean impPolicy = false;  // initializing the policy to be improved to false 
		double max = 0.0 ; //initializing the maximum value and the value to be calculated to 0
		double val = 0.0 ; 
		
		for (Game st : policyValues.keySet()) // iterating through the game states in policy values
		{
			if(st.isTerminal())  //if state is terminal state 
				continue;
			else
				max = Double.NEGATIVE_INFINITY;  // updating maximum value for non-terminal states 
			
			List<Move> pMoves = st.getPossibleMoves(); //assign the possible moves of state to variable 
			
			for(Move mv : pMoves) //iterate through moves
			{
				val = 0; // resetting value for each move 
				List<TransitionProb> tr = mdp.generateTransitions(st, mv);  // Retrieving transition probabilities 
				val += calcVal(tr); //Calculating the value 
				if (val > max) //if value greater than maximum value 
				{
					max = val; //updating  maximum value as calculated value 
					curPolicy.put(st, mv); // populating current policy with state and move 
					impPolicy = true; //Updating the policy to be improved as true
				}
			}
		}
		return impPolicy; // return the policy improved.
	}
	
	
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		/* YOUR CODE HERE */
		initValues();		// Initialing state values
		initRandomPolicy(); // Initializing random policy 
		
		do {				//iterating till convergence 
			evaluatePolicy(delta);  //Initializing the evaluate policy			
			HashMap<Game, Move> p = new HashMap<>(this.curPolicy);// Retrieving the current policy 
			improvePolicy();   // improving the policy
			if(curPolicy.equals(p) || !improvePolicy()) // if convergence or policy not improved break.
				break;
			
		}while(true); //iterate until until condition fails  
		policy = new Policy(); // Initializing new policy 
		policy.policy = curPolicy; // updating current policy as new policy 
		
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}

package ticTacToe;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=1000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()
	{
		this(new RandomAgent(), 0.1, 100, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 */
	
	public void train()
	{

		/* 
		 * YOUR CODE HERE
		 */					
		int totEp = numEpisodes * numEpisodes;  // calculating total training stages
		initQTable();                           // initializing Qtable

		for (int ep = 0; ep < totEp; ep++) { 	//Iterating over episodes
		    Game curSt = env.game;				// initializing current state

		    while (!curSt.isTerminal()) {
		        List<Move> posMv = curSt.getPossibleMoves(); //initializing the possible moves
		        Move sMv = null;							 //initializing current move

		        if (Math.random() < epsilon) {              // Exploration: choose a random action
		            if (!posMv.isEmpty()) {
		                int rIndex = new Random().nextInt(posMv.size());
		                sMv = posMv.get(rIndex);
		            }
		        } else {		            // Exploit: choose the action with the highest Q-value
		            double maxQval = Double.NEGATIVE_INFINITY;  //initializing maximum q value
		            List<Move> maxQmvs = new ArrayList<>();  //initializing list to store moves with maximum q values

		            for (Move mv : posMv) { //iterating through possible moves
		                double qVal = qTable.getQValue(curSt, mv);  //initialize qvalue for move

		                if (qVal > maxQval) {	// comparing if qvalue is greater than maximum value
		                    maxQval = qVal;
		                    maxQmvs.clear();
		                    maxQmvs.add(mv);	
		                } else if (qVal == maxQval) {
		                    maxQmvs.add(mv);
		                }
		            }

		            int randIndex = new Random().nextInt(maxQmvs.size());  //generating random index
		            sMv = maxQmvs.get(randIndex);				//initializing move with highest-qvalue randomly
		        }

		        Outcome otC = null;		//initializing output
		        try {
		            otC = env.executeMove(sMv);  //executing the current move
		        } catch (IllegalMoveException err) { //handling error
		            err.printStackTrace();
		        }

		        Game srcSt = otC.s;			//source of output
		        Game trgSt = otC.sPrime;	//target of output

		        List<Move> trgPosMoves = trgSt.getPossibleMoves(); //initializing list to store possible moves in target
		        double maxTrgQVal = Double.NEGATIVE_INFINITY;		//initializing maximum qvalue in target

		        for (Move trgMv : trgPosMoves) {  //iterating through possible moves
		            double trgQVal = qTable.getQValue(trgSt, trgMv); //initializing qvalue of move
		            if (trgQVal > maxTrgQVal) { //updating maximum value if qvalue is greater
		                maxTrgQVal = trgQVal;
		            }
		        }

		        if (trgSt.isTerminal()) {  //updating maximum value as o if target is terminal
		            maxTrgQVal = 0.0;
		        }

		        double nQVal = (((1-alpha)*(qTable.getQValue(srcSt, sMv)))+(alpha*((otC.localReward)+(discount*maxTrgQVal))));  //calculating new qvalue

		        qTable.addQValue(srcSt, sMv, nQVal);  //updating qvalue
		        curSt = otC.sPrime;
		    }

		    env.reset();
		}

		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	}
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		/* 
		 * YOUR CODE HERE
		 */	
	    Policy exPol = new Policy();			// Initializing policy object
	    
	    for (Game state : qTable.keySet()) {		// Iterating through states from Qtable 
	        HashMap<Move, Double> aVal = qTable.get(state); // assigning move and qValue of state to variable

	        Move bMove = null;					//Initializing best Move 
	        double bVal = Double.NEGATIVE_INFINITY;		//Initializing best value 

	        for(Move mv : aVal.keySet()) {  //Iterating through moves of the variable
	            double val = aVal.get(mv);// assigning vale from move 

	            if (val > bVal)  { 	// if value is greater than best value 
	                bVal = val;  	// Updating best value as calculated value 
	                bMove = mv;    // Updating best move as current move 
	            }
	        }

        if (bMove != null) {  	// if best move not null
	            exPol.policy.put(state, bMove); 	// updating policy with state and best move 
	        }
	    }

	    return exPol; // return policy 

	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
	
	
	


	
}

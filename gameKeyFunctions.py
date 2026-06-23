import math
import copy
import random
import os
import bisect
from constants import depthProbabilityMultiplier
import vizdoom as vzd
from enum import Enum, auto
import gameLogic
def resetGame(showGame):
    game = vzd.DoomGame()
    config = {
        "doom_scenario_path": os.path.join(os.getcwd(), "deathmatch3.wad.backup1"),
        "doom_map": "map01",
        "doom_skill": 2,

        "available_buttons": [
            vzd.Button.ATTACK,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.MOVE_FORWARD,
            vzd.Button.TURN_RIGHT,
            vzd.Button.TURN_LEFT,
            vzd.Button.SELECT_WEAPON3,
            vzd.Button.SELECT_WEAPON4
        ],

        # MUST enable richer state
        "available_game_variables": [
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.ARMOR,
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.ANGLE,
            vzd.GameVariable.KILLCOUNT,
            vzd.GameVariable.SELECTED_WEAPON,
            vzd.GameVariable.AMMO3,
            vzd.GameVariable.AMMO4
        ],

        "episode_timeout": 4200,
        "episode_start_time": 1,
        "living_reward": 0.01,
        "mode": vzd.Mode.PLAYER,
        "screen_resolution": vzd.ScreenResolution.RES_1920X1080 if showGame else vzd.ScreenResolution.RES_160X120,
        "render_hud": showGame,        
    }

    game.set_config(config)
    game.set_objects_info_enabled(True)
    game.set_window_visible(showGame)

    game.init()
    return game

allConditions = {
    "lowHealth?": gameLogic.lowHealthCondition,
    "mediumHealth?": gameLogic.mediumHealthCondition,
    "highHealth?": gameLogic.highHealthCondition,
    "lowArmor?": gameLogic.lowArmorCondition,
    "mediumArmor?": gameLogic.mediumArmorCondition,
    "highArmor?": gameLogic.highArmorCondition,
    "lowAmmoCurrent?": gameLogic.lowAmmoCurrent,
    "chaingunEquipped?": gameLogic.wieldingChaingun,
    "nearbyEnemy?": gameLogic.nearbyEnemy,
    "recentlyHurt?": gameLogic.recentlyHurt,
    "healthNearby?": gameLogic.healthNearby,
    "ammo3Nearby?": gameLogic.ammo3Nearby,
    "ammo4Nearby?": gameLogic.ammo4Nearby,
    "armorNearby?": gameLogic.armorNearby,
    "manyEnemies?": gameLogic.manyEnemies,
    "noEnemies?":gameLogic.noEnemies,
    "someRangedEnemy?": gameLogic.someRangedEnemies,
    "lowTimeRemaining?": gameLogic.lowTimeRemaining
}

allActions = {
    "switchWeapon": gameLogic.SwitchWeapon,
    "fireAndStrafe": gameLogic.FireAndStrafe,
    "directlyFlee": gameLogic.DirectlyFlee,
    "goToHealth": gameLogic.GoToHealth,
    "goToAmmo": gameLogic.GoToAmmo,
    "goToArmor": gameLogic.GoToArmor,
    "moveRandom": gameLogic.MoveRandom,
    "runAway": gameLogic.RunAway,
    "chargeIn": gameLogic.ChargeIn
}

class NodeType(Enum):
    ACTION = auto()
    CONDITION = auto()

class Node():
    def __init__(self, depth, parent, nodeType, overallTree, name, mainHandler):
        self.depth = depth
        self.parent = parent
        self.type = nodeType
        self.protected = False
        self.tree = overallTree

        self.left = None
        self.right = None

        self.nodeName = name
        self.nodeHandler = mainHandler() if self.type == NodeType.ACTION else mainHandler

    def mutate(self):
        while True:
            chosenList = list(allConditions.keys()) if self.type == NodeType.CONDITION else list(allActions.keys())

            chosenIndex = random.randint(0, len(chosenList) - 1)
            newHandlerName = chosenList[chosenIndex]

            if(newHandlerName == self.nodeName):
                continue
            
            self.nodeName = newHandlerName
            self.nodeHandler = allConditions[self.nodeName] if self.type == NodeType.CONDITION else allActions[self.nodeName]()
            break
            
        


class Tree():
    def __init__(self, maxDepth, probabilityOfEndingEarly):
        self.maxDepth = maxDepth
        self.probabilityOfEndingEarly = probabilityOfEndingEarly
        self.currentAction = None
        self.allNodes = set()


    def buildNode(self, conditions, actions, currentDepth, parentNode, conditionsInPath):
        randomProbability = random.random()
        if(currentDepth == self.maxDepth or (currentDepth != 0 and randomProbability <= self.probabilityOfEndingEarly)):
            nodeType = NodeType.ACTION
        else:
            nodeType = NodeType.CONDITION
        

        chosenListToSelect = conditions if nodeType == NodeType.CONDITION else actions
        chosenElement = None
        while True:
            chosenElement = random.choice(list(chosenListToSelect.keys()))
            if(chosenElement not in conditionsInPath):
                break
        newNode = Node(currentDepth, parentNode, nodeType, self, chosenElement, chosenListToSelect[chosenElement])
        
        if(nodeType == NodeType.CONDITION):
            conditionsInPath.add(chosenElement)
            newNode.left = self.buildNode(conditions, actions, currentDepth + 1, newNode, conditionsInPath.copy())
            newNode.right = self.buildNode(conditions, actions, currentDepth + 1, newNode, conditionsInPath.copy())

        self.allNodes.add(newNode)

        return newNode    

    def decideNewAction(self, state, currentTick, currentNode):
        if(currentNode.type == NodeType.ACTION):
            self.currentAction = currentNode
            self.currentAction.nodeHandler.activateAction(state, currentTick)
            return
        
        if(currentNode.nodeHandler(state, currentTick)):
            return self.decideNewAction(state, currentTick, currentNode.left)

        return self.decideNewAction(state, currentTick, currentNode.right)

    def updateActionTick(self, state, currentTick):
        action, reward = self.currentAction.nodeHandler.updateTickAndReturnAction(state, currentTick)
        if(self.currentAction.nodeHandler.finished):
            self.currentAction = None
        return action, reward 

def BuildRandomTree(maxDepth, probEndingEarly, conditions, actions):
    tree = Tree(maxDepth, probEndingEarly)
    tree.root = tree.buildNode(conditions, actions, 0, None, set())
    return tree

def RankTreesByScore(treeArray):
    return sorted(treeArray, key = lambda x : x.score)

def clearNodesRecursively(currentNode):
    currentNode.protected = False
    
    if(currentNode.type == NodeType.ACTION):
        return
    clearNodesRecursively(currentNode.left)
    clearNodesRecursively(currentNode.right)

def DoTreeUpkeep(givenTree):
    givenTree.score = 0
    clearNodesRecursively(givenTree.root)

def ShowTreeStructure(node, indent=""):
    if node.type == NodeType.ACTION:
        print(indent + "[ACTION] " + node.nodeName)
        return

    print(indent + "[COND] " + node.nodeName)

    print(indent + "  T:")
    ShowTreeStructure(node.left, indent + "    ")

    print(indent + "  F:")
    ShowTreeStructure(node.right, indent + "    ")

def RunGame(behaviorTree, game, episodes):
    global previousHealth, lastTickHurtCalled
    score = 0
    for episode in range(episodes):
        game.new_episode()
        gameLogic.previousHealth = 100
        gameLogic.lastTickHurtCalled = -200

        tick = 0

        print(f"score so far: {score}")
        epScore = 0
        while not game.is_episode_finished():
            state = game.get_state()
            if behaviorTree.currentAction is None:
                behaviorTree.decideNewAction(state, tick, behaviorTree.root)
                # print("New Action gotten!")

            # print(f"CURRENT ACTION IS: {behaviorTree.currentAction.nodeName}")

            chosenAction, anyRewardPunishment = behaviorTree.updateActionTick(state, tick)

            epScore += anyRewardPunishment

            epScore += game.make_action(chosenAction) / 100.0

            health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables

            if(health > gameLogic.previousHealth):
                gameLogic.previousHealth = health
            elif(health < gameLogic.previousHealth):
                gameLogic.previousHealth = health
                gameLogic.lastTickHurtCalled = tick

            tick += 1

            if(tick == 4200):
                break

        score += epScore / episodes

    return score

def MutateTrees(givenTrees, numToCreate):
    createdTrees = []
    
    totalScore = 0
    scoreArrForBinarySearch = []
    correspondingTree = []

    minScore = min(tree.score for tree in givenTrees)
    shift = -minScore if minScore < 0 else 0

    for tree in givenTrees:
        totalScore += tree.score + shift
        scoreArrForBinarySearch.append(totalScore)
        correspondingTree.append(tree)
    
    for _ in range(numToCreate):
        numberInArr = random.uniform(0, totalScore)
        foundIndex = bisect.bisect_left(scoreArrForBinarySearch, numberInArr)
        chosenTree = copy.deepcopy(correspondingTree[foundIndex])

        nodesAsList = list(chosenTree.allNodes)
        nodeNum = random.randint(0, len(nodesAsList) - 1)
        theActualNode = nodesAsList[nodeNum]

        theActualNode.mutate()

        createdTrees.append(chosenTree)

    return createdTrees

def CrackedCanonicalRep(node):
    if node == None:
        return
    
    return (node.nodeName, CrackedCanonicalRep(node.left), CrackedCanonicalRep(node.right))
    

def ExploreTreeAndFindMatches(currentNode, treesSeenThusFarCount, treesToNodesInvolvedMapping, representationsToAddCurrentNodeTo):
    if(currentNode.type == NodeType.ACTION):
        return
    
    currentNodeRepresentation = CrackedCanonicalRep(currentNode)
    if currentNodeRepresentation not in treesSeenThusFarCount:
        treesSeenThusFarCount[currentNodeRepresentation] = 1
    else:
        treesSeenThusFarCount[currentNodeRepresentation] += 1
    
    for element in representationsToAddCurrentNodeTo:
        if element not in treesToNodesInvolvedMapping:
            treesToNodesInvolvedMapping[element] = [currentNode]
        else:
            treesToNodesInvolvedMapping[element].append(currentNode)

    representationsToAddCurrentNodeTo.append(currentNodeRepresentation)

    ExploreTreeAndFindMatches(currentNode.left, treesSeenThusFarCount, treesToNodesInvolvedMapping, representationsToAddCurrentNodeTo)
    ExploreTreeAndFindMatches(currentNode.right, treesSeenThusFarCount, treesToNodesInvolvedMapping, representationsToAddCurrentNodeTo)

def ProtectCommonTrees(trees, protectCount):
    treesSeenThusFarCount = {}
    treesToNodesInvolvedMapping = {}

    for tree in trees:
        ExploreTreeAndFindMatches(tree.root, treesSeenThusFarCount, treesToNodesInvolvedMapping, [])

    listedTreesVersion = list(treesSeenThusFarCount.keys())
    for key in listedTreesVersion:
        if treesSeenThusFarCount[key] < protectCount:
            del treesSeenThusFarCount[key]
            if key in treesToNodesInvolvedMapping:
                del treesToNodesInvolvedMapping[key]
    
    return treesToNodesInvolvedMapping.values()

def UpdateDepths(currentNode, newDepth):
    if currentNode == None:
        return
    
    currentNode.depth = newDepth
    UpdateDepths(currentNode.left, newDepth + 1)
    UpdateDepths(currentNode.right, newDepth + 1)

def PerformCrossover(node1, node2):
    node1Parent = node1.parent
    node2Parent = node2.parent
    if(node1Parent.left == node1):
        if(node2Parent.left == node2):
            node1Parent.left = node2
            node2Parent.left = node1
        else:
            node1Parent.left = node2
            node2Parent.right = node1
    else:
        if(node2Parent.left == node2):
            node1Parent.right = node2
            node2Parent.left = node1
        else:
            node1Parent.right = node2
            node2Parent.right = node1

    UpdateDepths(node1, node2Parent.depth + 1)
    UpdateDepths(node2, node1Parent.depth + 1)

    return (node1.tree, node2.tree)


def CrossoverTrees(givenTrees, numToCreate, countToBeProtected):
    protectedNodes = ProtectCommonTrees(givenTrees, countToBeProtected)
    createdTrees = []

    for i in range(math.floor(numToCreate / 2)):
        maxNumTries = 0
        
        chosenTreeA = copy.deepcopy(random.choice(givenTrees))

        chosenNodeA = None
        escaping = False
        while True:
            chosenNodeA = random.choice(list(chosenTreeA.allNodes))
            if(chosenNodeA.type != NodeType.ACTION and chosenNodeA != chosenTreeA.root):
                if(chosenNodeA in protectedNodes):
                    chanceOfAccepting = random.random()
                    if(chanceOfAccepting >= 0.5):
                        maxNumTries += 1
                        if(maxNumTries >= 100):
                            escaping = True
                            break
                        continue
                break
            maxNumTries += 1
            if(maxNumTries >= 100):
                escaping = True
                break
        
        if escaping:
            continue
        maxNumTries = 0
        chosenTreeB = chosenTreeA
        while chosenTreeA == chosenTreeB:
            chosenTreeB = copy.deepcopy(random.choice(givenTrees))

        chosenNodeB = None
        while True:
            chosenNodeB = random.choice(list(chosenTreeB.allNodes))
            if(chosenNodeB.type != NodeType.ACTION and chosenNodeB != chosenTreeB.root and CrackedCanonicalRep(chosenNodeA) != CrackedCanonicalRep(chosenNodeB)):
                if(chosenNodeB in protectedNodes):
                    chanceOfAccepting = random.random()
                    if(chanceOfAccepting >= 0.5):
                        maxNumTries += 1
                        if(maxNumTries >= 100):
                            escaping = True
                            break
                        continue
                        
                depth = abs(chosenNodeA.depth - chosenNodeB.depth)
                chanceOfAccepting = random.random()
                if(chanceOfAccepting >= min(depth * depthProbabilityMultiplier, 0.8)):
                    break 
            maxNumTries += 1
            if(maxNumTries >= 100):
                escaping = True
                break
        if escaping:
            continue                      
        
        chosenTreeA, chosenTreeB = PerformCrossover(chosenNodeA, chosenNodeB)
        createdTrees.append(chosenTreeA)
        createdTrees.append(chosenTreeB)
    
    return createdTrees

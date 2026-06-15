nonEnemyObjects = ["Stimpack", "Medikit", "HealthBonus", "ClipBox", "DoomPlayer", "BlueArmor", "ShellBox", "TeleportFog", "BaronBall", "BulletPuff", "Blood"]
distanceUntilTooClose = 200
ticksFor5Seconds = 175
lastTickHurtCalled = -200
previousHealth = 100
tickOfEnd = 4200

def lowHealthCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return health <= 33.3

def mediumHealthCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return health > 33 and health <= 66.6

def highHealthCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return health > 66.6

def lowArmorCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return armor <= 33.3

def mediumArmorCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return armor > 33 and armor <= 66.6

def highArmorCondition(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return armor > 66.6

def lowAmmoCurrent(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # if super shotgun
    if(currentWeapon == 3):
        return firstWepAmmo < 10
    # otherwise, chaingun
    return secondWepAmmo < 40

def wieldingChaingun(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    return currentWeapon == 4

def nearbyEnemy(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name not in nonEnemyObjects):
           playerDistance = ((object.position_x - posX)**2 + (object.posiition_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def recentlyHurt(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    if(health < previousHealth):
        previousHealth = health
        if(currentTick - lastTickHurtCalled < ticksFor5Seconds):
            lastTickHurtCalled = currentTick
            return True
    lastTickHurtCalled = currentTick
    return False

def healthNearby(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name == "Medikit"):
           playerDistance = ((object.position_x - posX)**2 + (object.posiition_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def ammo3Nearby(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name == "ShellBox"):
           playerDistance = ((object.position_x - posX)**2 + (object.posiition_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def ammo4Nearby(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name == "ClipBox"):
           playerDistance = ((object.position_x - posX)**2 + (object.posiition_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def armorNearby(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    for object in state.objects:
       if(object.name == "BlueArmor"):
           playerDistance = ((object.position_x - posX)**2 + (object.posiition_y - posY) **2) ** (1/2)
           if(playerDistance < distanceUntilTooClose):
               return True
    
    return False

def manyEnemies(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    count = 0
    for object in state.objects:
       if(object.name not in nonEnemyObjects):
           count += 1
    
    return count >= 6

def noEnemies(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    count = 0
    for object in state.objects:
       if(object.name not in nonEnemyObjects):
           count += 1
    
    return count == 0

def someRangedEnemies(state, currentTick):
    health, armor, posX, posY, angle, kills, currentWeapon, firstWepAmmo, secondWepAmmo = state.game_variables
    # TO DO: REST OF THIS
    # ideal distance is probably 200
    numRanged = 0
    for object in state.objects:
       if(object.name == "ChaingunGuy" or object.name == "HellKnight"):
           numRanged += 1
    
    return numRanged >= 2

def lowTimeRemaining(state, currentTick):
    return currentTick >= (tickOfEnd - 500)
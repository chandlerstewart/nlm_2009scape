package python;

import core.game.bots.AIPlayer;

import java.util.HashMap;

public class Action {

    public static void move(AIPlayer bot, HashMap botMap){
        int xLoc = ((Double) botMap.get("xLoc")).intValue();
        int yLoc = ((Double) botMap.get("yLoc")).intValue();
        bot.walkToPosSmart(xLoc, yLoc);
    }
}

package python;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import core.game.bots.AIPlayer;
import core.game.node.Node;
import core.game.world.map.Direction;
import core.game.world.map.Location;
import core.game.world.map.RegionManager;

import java.util.ArrayList;
import java.util.HashMap;

public class PyBotManager {

    public static ArrayList<AIPlayer> botList = new ArrayList<AIPlayer>();
    public static Integer nodeRange = 0;

    public static String getJSONBotStatus(){

        ArrayList<JsonElement> botInfoList = new ArrayList<JsonElement>();


        for (int i=0;i<botList.size();i++){
            AIPlayer bot = botList.get(i);
            String name = bot.getName();
            int xLoc = bot.getLocation().getX();
            int yLoc = bot.getLocation().getY();


            BotInfo botInfo = new BotInfo();
            botInfo.put("xLoc", xLoc);
            botInfo.put("yLoc", yLoc);
            botInfo.put("name", name);

            botInfo.put("freeInvSpace", bot.getInventory().freeSlots());

            //addNearbyNodesToBotInfo(bot, botInfo);
            botInfo.put("nearbyNodes",bot.getNodesInRange(nodeRange));

            botInfoList.add(botInfo.toJsonElement());
        }

        Gson gson = new Gson();
        String jsonString = gson.toJson(botInfoList);
        return jsonString;
    }


    private static void addNodeHelper(Node node, String nodeName, BotInfo botInfo){
        if (node != null){
            botInfo.put(nodeName,true);
        } else {
            botInfo.put(nodeName,false);
        }
    }


    public static void process(ArrayList<BotInfo> botInfoList){
        for(int i=0; i<botInfoList.size(); i++){

            AIPlayer bot = botList.get(i);
            HashMap botMap = botInfoList.get(i).map;

            String action = (String) botMap.get("action");
            switch(action){
                case "move":
                    Action.move(bot, botMap);
                    break;
            }

        }

    }




    public static void takeActions(ArrayList<BotInfo> botInfoList) {
        for(int i=0; i<botInfoList.size(); i++){

            AIPlayer bot = botList.get(i);
            HashMap botMap = botInfoList.get(i).map;

            int xLoc = ((Double) botMap.get("xLoc")).intValue();
            int yLoc = ((Double) botMap.get("yLoc")).intValue();

            if (botMap.get("interact").equals("none")) {
                bot.walkToPosSmart(xLoc, yLoc);
            }
        }

    }

    public static void removeBots(){
        for (AIPlayer aiPlayer : botList){
            aiPlayer.clear();
        }

        botList.clear();
    }

    private static int boolToInt(boolean boolVal){
        return boolVal? 1 : 0;
    }


}



package python;

//import rt4.*;

import core.game.bots.AIPlayer;
import core.game.container.impl.EquipmentContainer;
import core.game.node.entity.skill.Skills;
import core.game.node.item.Item;
import core.game.world.map.Location;
import org.rs09.consts.Items;

import java.util.ArrayList;

public class PyCommands {

    public static Message ProcessCommand(Message message){


        String command = message.getCommand();
        String[] split = message.getCommand().split(" ");
        ArrayList<BotInfo> botInfoList;

        System.out.println(command);


        switch(split[0]){
            case "spawn_bots":

                int x = Integer.valueOf(split[1]);
                int y = Integer.valueOf(split[2]);
                int numOfBots = Integer.valueOf(split[3]);
                botInfoList = message.getInfo();
                GenerateBot(new Location(x, y), numOfBots, botInfoList);
                String jsonBotStatus = PyBotManager.getJSONBotStatus();

                return new Message("Success: spawn_bots", String.valueOf(jsonBotStatus.length()*2));
            case "server_waiting":
                return new Message("json", PyBotManager.getJSONBotStatus());
            case "json":
                botInfoList = message.getInfo();
                PyBotManager.process(botInfoList);
                PyBotManager.takeActions(botInfoList);
                return new Message("json", PyBotManager.getJSONBotStatus());

            default:
                System.out.println("DEFAULT");
                return new Message("", "");

        }
    }

    private static void GenerateBot(Location location, int numOfBots, ArrayList<BotInfo> botInfoList){

        String task = (String) botInfoList.get(0).map.get("task");
        Integer nodeRange = ((Double) botInfoList.get(0).map.get("nodesRange")).intValue();
        PyBotManager.nodeRange = nodeRange;

        if (PyBotManager.botList.size() > 0){
            PyBotManager.removeBots();
        }


        for (int i=0; i<numOfBots; i++){
            AIPlayer aiPlayer = new AIPlayer("Bot" + i, location, "");

            PyBotManager.botList.add(aiPlayer);

        }
    }

    }


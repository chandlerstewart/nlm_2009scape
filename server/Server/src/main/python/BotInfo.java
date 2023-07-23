package python;


import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.HashMap;

class BotInfo {

    public HashMap<String, Object> map;

    BotInfo(){
        map = new HashMap<>();
    }

    BotInfo(HashMap map){
        this.map = map;
    }

    public void put(String key, Object value){
        map.put(key,value);
    }

    public JsonElement toJsonElement(){
        Gson gson = new Gson();
        JsonElement jsonElement = gson.toJsonTree(map);
        return jsonElement;
    }

    public String toJsonString(){
        Gson gson = new Gson();
        JsonElement jsonElement = toJsonElement();
        JsonObject jsonObject = jsonElement.getAsJsonObject();
        String jsonString = gson.toJson(jsonObject);
        return jsonString;
    }

    public static ArrayList<BotInfo> mapToBotInfo(ArrayList<HashMap<String, Object>> listOfMaps){
        ArrayList<BotInfo> botInfoList = new ArrayList<>();
        for (HashMap map : listOfMaps){
            botInfoList.add(new BotInfo(map));
        }

        return botInfoList;
    }
}
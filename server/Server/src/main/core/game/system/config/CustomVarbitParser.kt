package core.game.system.config

import core.cache.def.impl.VarbitDefinition
import org.json.simple.JSONArray
import org.json.simple.JSONObject
import org.json.simple.parser.JSONParser
import core.ServerConstants
import core.api.log
import core.tools.Log
import core.tools.SystemLogger
import java.io.FileReader

class CustomVarbitParser {
    val parser = JSONParser()
    var reader: FileReader? = null

    fun load() {
        var count = 0

        reader = FileReader(ServerConstants.CONFIG_PATH + "varbit_definitions.json")
        val array = parser.parse(reader) as JSONArray

        for (jObject in array) {
            val varbitDef = jObject as JSONObject

            val varpId = varbitDef["varpId"].toString().toInt()
            val varbitId = varbitDef["varbitId"].toString().toInt()
            val startBit = varbitDef["startBit"].toString().toInt()
            val endBit = varbitDef["endBit"].toString().toInt()

            VarbitDefinition.create(varpId, varbitId, startBit, endBit)
            count++
        }

        log(this::class.java, Log.FINE,  "Parsed $count custom varbit definitions.")
    }
}
package content.global.skill.farming

import core.Util.clamp
import core.game.node.entity.player.Player
import core.game.system.task.Pulse
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.json.simple.JSONArray
import org.json.simple.JSONObject
import core.game.node.entity.state.PlayerState
import core.game.node.entity.state.State
import core.tools.SystemLogger
import java.util.concurrent.TimeUnit

@PlayerState("farming")
class FarmingState(player: Player? = null) : State(player) {
    private val patchMap = HashMap<FarmingPatch, Patch>()
    private val binMap = HashMap<CompostBins, CompostBin>()


    fun getPatch(patch: FarmingPatch): Patch {
        return patchMap[patch] ?: (Patch(player!!,patch).also { patchMap[patch] = it })
    }

    fun getBin(bin: CompostBins) : CompostBin{
        return binMap[bin] ?: (CompostBin(player!!,bin).also { binMap[bin] = it })
    }

    fun getPatches(): MutableCollection<Patch>{
        return patchMap.values
    }

    fun getBins(): MutableCollection<CompostBin>{
        return binMap.values
    }

    override fun save(root: JSONObject) {
        val patches = JSONArray()
        for((key,patch) in patchMap){
            val p = JSONObject()
            p.put("patch-ordinal",key.ordinal)
            p.put("patch-plantable-ordinal",patch.plantable?.ordinal ?: -1)
            p.put("patch-watered",patch.isWatered)
            p.put("patch-diseased",patch.isDiseased)
            p.put("patch-dead",patch.isDead)
            p.put("patch-stage",patch.currentGrowthStage)
            p.put("patch-state",patch.getCurrentState())
            p.put("patch-nextGrowth",patch.nextGrowth)
            p.put("patch-harvestAmt",patch.harvestAmt)
            p.put("patch-checkHealth",patch.isCheckHealth)
            p.put("patch-compost",patch.compost.ordinal)
            p.put("patch-paidprot",patch.protectionPaid)
            p.put("patch-croplives", patch.cropLives)
            patches.add(p)
        }
        val bins = JSONArray()
        for((key,bin) in binMap){
            val b = JSONObject()
            b.put("bin-ordinal",key.ordinal)
            bin.save(b)
            bins.add(b)
        }
        root.put("farming-patches",patches)
        root.put("farming-bins",bins)
    }

    override fun parse(_data: JSONObject) {
        player ?: return
        if(_data.containsKey("farming-bins")){
            (_data["farming-bins"] as JSONArray).forEach {
                val bin = it as JSONObject
                val binOrdinal = bin["bin-ordinal"].toString().toInt()
                val cBin = CompostBins.values()[binOrdinal]
                val b = cBin.getBinForPlayer(player)
                b.parse(bin["binData"] as JSONObject)
            }
        }
        if(_data.containsKey("farming-patches")){
            val data = _data["farming-patches"] as JSONArray
            for(d in data){
                val p = d as JSONObject
                val patchOrdinal = p["patch-ordinal"].toString().toInt()
                val patchPlantableOrdinal = p["patch-plantable-ordinal"].toString().toInt()
                val patchWatered = p["patch-watered"] as Boolean
                val patchDiseased = p["patch-diseased"] as Boolean
                val patchDead = p["patch-dead"] as Boolean
                val patchStage = p["patch-stage"].toString().toInt()
                val nextGrowth = p["patch-nextGrowth"].toString().toLong()
                val harvestAmt = (p["patch-harvestAmt"] ?: 0).toString().toInt()
                val checkHealth = p["patch-checkHealth"] as Boolean
                val savedState = p["patch-state"].toString().toInt()
                val compostOrdinal = p["patch-compost"].toString().toInt()
                val protectionPaid = p["patch-paidprot"] as Boolean
                val cropLives = if(p["patch-croplives"] != null) p["patch-croplives"].toString().toInt() else 3
                val fPatch = FarmingPatch.values()[patchOrdinal]
                val plantable = if(patchPlantableOrdinal != -1) Plantable.values()[patchPlantableOrdinal] else null
                val patch = Patch(player,fPatch,plantable,patchStage,patchDiseased,patchDead,patchWatered,nextGrowth,harvestAmt,checkHealth)

                patch.cropLives = cropLives
                patch.compost = CompostType.values()[compostOrdinal]
                patch.protectionPaid = protectionPaid
                patch.setCurrentState(savedState)

                if((savedState - (patch?.plantable?.value ?: 0)) > patch.currentGrowthStage){
                    patch.setCurrentState(savedState)
                } else {
                    patch.setCurrentState((patch.plantable?.value ?: 0) + patch.currentGrowthStage)
                }

                val type = patch.patch.type
                val shouldPlayCatchup = !patch.isGrown() || (type == PatchType.BUSH && patch.getFruitOrBerryCount() < 4) || (type == PatchType.FRUIT_TREE && patch.getFruitOrBerryCount() < 6)
                if(shouldPlayCatchup && patch.plantable != null && !patchDead){
                    var stagesToSimulate = if (!patch.isGrown()) patch.plantable!!.stages - patch.currentGrowthStage else 0
                    if (type == PatchType.BUSH)
                        stagesToSimulate += Math.min(4, 4 - patch.getFruitOrBerryCount())
                    if (type == PatchType.FRUIT_TREE)
                        stagesToSimulate += Math.min(6, 6 - patch.getFruitOrBerryCount())

                    val nowTime = System.currentTimeMillis()
                    var simulatedTime = patch.nextGrowth

                    while (simulatedTime < nowTime && stagesToSimulate-- > 0 && !patch.isDead) {
                        val timeToIncrement = TimeUnit.MINUTES.toMillis(patch.getStageGrowthMinutes().toLong())
                        patch.update()
                        simulatedTime += timeToIncrement
                    }
                }

                if(patchMap[fPatch] == null) {
                    patchMap[fPatch] = patch
                }
            }
        }
    }

    override fun newInstance(player: Player?): State {
        return FarmingState(player)
    }

    override fun createPulse() {
        pulse = object : Pulse(3){
            override fun pulse(): Boolean {

                GlobalScope.launch {
                    var removeList = ArrayList<FarmingPatch>()
                    for((_,patch) in patchMap){

                        if(patch.getCurrentState() in 1..3 && patch.nextGrowth == 0L){
                            patch.nextGrowth = System.currentTimeMillis() + 60000
                            continue
                        }

                        if(patch.nextGrowth < System.currentTimeMillis() && !patch.isDead){
                            patch.update()
                            patch.nextGrowth = System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(patch.getStageGrowthMinutes().toLong())
                        }

                    }

                    for((_,bin) in binMap){
                        if(bin.isReady() && !bin.isFinished){
                            bin.finish()
                        }
                    }
                }

                return false
            }
        }
    }
}

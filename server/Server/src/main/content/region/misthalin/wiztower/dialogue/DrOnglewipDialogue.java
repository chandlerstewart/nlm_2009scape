package content.region.misthalin.wiztower.dialogue;

import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.node.entity.npc.NPC;
import core.plugin.Initializable;
import core.game.node.entity.player.Player;

/**
 * Represents the dr onglewip dialogue plugin.
 * @author 'Vexia
 * @version 1.0
 */
@Initializable
public final class DrOnglewipDialogue extends DialoguePlugin {

	/**
	 * Constructs a new {@code DrOnglewipDialogue} {@code Object}.
	 */
	public DrOnglewipDialogue() {
		/**
		 * empty.
		 */
	}

	/**
	 * Constructs a new {@code DrOnglewipDialogue} {@code Object}.
	 * @param player the player.
	 */
	public DrOnglewipDialogue(Player player) {
		super(player);
	}

	@Override
	public DialoguePlugin newInstance(Player player) {
		return new DrOnglewipDialogue(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		interpreter.sendDialogues(player, FacialExpression.ASKING, "Do you live here too?");
		stage = 0;
		return true;
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {
		switch (stage) {
		case 0:
			interpreter.sendDialogues(npc, FacialExpression.OLD_NORMAL, "Oh no, I come from Gnome Stronghold. I've been", "sent here by King Narnode to learn about human", "magics.");
			stage = 1;
			break;
		case 1:
			interpreter.sendDialogues(player, FacialExpression.ASKING, "So where's this Gnome Stronghold?");
			stage = 2;
			break;
		case 2:
			interpreter.sendDialogues(npc, FacialExpression.OLD_NORMAL, "It's in the North West of the continent - a long way", "away. You should visit us there some time. The food's", "great, and the company's delightful.");
			stage = 3;
			break;
		case 3:
			interpreter.sendDialogues(player, FacialExpression.FRIENDLY, "I'll try and make time for it. Sounds like a nice place.");
			stage = 4;
			break;
		case 4:
			interpreter.sendDialogues(npc, FacialExpression.OLD_NORMAL, "Well, it's full of gnomes. How much nicer could it be?");
			stage = 5;
			break;
		case 5:
			end();
			break;
		}
		return true;
	}

	@Override
	public int[] getIds() {
		return new int[] { 4585 };
	}
}

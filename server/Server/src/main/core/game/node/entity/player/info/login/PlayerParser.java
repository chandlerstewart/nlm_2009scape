package core.game.node.entity.player.info.login;

import core.game.node.entity.player.Player;
import core.ServerConstants;
import core.tools.Log;
import core.tools.SystemLogger;

import java.io.*;

import static core.api.ContentAPIKt.log;

/**
 * Class used to abstract the process of loading a player save.
 * @author Ceikry
 */
public final class PlayerParser {
	/**
	 * Parses or creates the player's save file depending on whether or not it exists.
	 * @param player The player.
	 */
	public static PlayerSaveParser parse(Player player) {
		PlayerSaveParser parser = new PlayerSaveParser(player);

		try {
			parser.parse();
			return parser;
		} catch (Exception e){
			e.printStackTrace();
			return null;
		}
	}
	/**
	 * Saves the player's details to the character file at data/players/player_name.json
	 * @param player The player.
	 */
	public static void save(Player player) {
		player.setAttribute("flagged-for-save", true);
	}

	public static void saveImmediately(Player player) {
		new PlayerSaver(player).save();
	}

	/**
	 * Copies the template at data/players/template/template.json to data/players/player_name.json
	 * @param player the player to copy the template for.
	 */
	public static void makeFromTemplate(Player player){
		InputStream is = null;
		OutputStream os = null;
		try {
			is = new FileInputStream(ServerConstants.PLAYER_SAVE_PATH + "template/template.json");
			os = new FileOutputStream(ServerConstants.PLAYER_SAVE_PATH + player.getName() + ".json");
			byte[] buffer = new byte[1024];
			int length;
			while ((length = is.read(buffer)) > 0) {
				os.write(buffer, 0, length);
			}
		} catch (Exception e){ e.printStackTrace();
		} finally {
			try {
				assert is != null;
				is.close();
				assert os != null;
				os.close();
			} catch (Exception f){
				log(PlayerParser.class, Log.ERR, "Unable to close file copiers in PlayerParser.java line 216.");
				f.printStackTrace();
			}
		}
	}
}
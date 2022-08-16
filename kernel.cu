// cardiac-cell-solver.cpp : Este arquivo contém a função 'main'. A execução do programa começa e termina ali.
//


#include <string>
#ifdef  OSisWindows
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\cuda_runtime.h" // RASCUNHO
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include\device_launch_parameters.h"
#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#endif
#include <vector>
#include <map>
#include <list>
#include <iostream>
#include <fstream>
#include <sstream>

#include "defs.h"






namespace CommandLineProcessing
{

	class ArgvParser
	{
	public:
		typedef int OptionAttributes;
		typedef int ParserResults;
		typedef std::map<std::string, unsigned int> String2KeyMap;
		typedef std::map<unsigned int, OptionAttributes> Key2AttributeMap;
		typedef std::map<unsigned int, std::string> Key2StringMap;
		typedef std::vector<std::string> ArgumentContainer;

		ArgvParser();
		~ArgvParser();

		/** Attributes for options. */
		enum
		{
			NoOptionAttribute = 0x00,
			OptionRequiresValue = 0x01,
			OptionRequired = 0x02
		};
		/** Return values of the parser. */
		enum
		{
			NoParserError = 0x00,
			ParserUnknownOption = 0x01,
			ParserMissingValue = 0x02,
			ParserOptionAfterArgument = 0x04,
			ParserMalformedMultipleShortOption = 0x08,
			ParserRequiredOptionMissing = 0x16,
			ParserHelpRequested = 0x32
		};

		/** Defines an option with optional attributes (required, ...) and an
		* additional (also optional) description. The description becomes part of the
		* generated usage help that can be requested by calling the usageDescription()
		* method.
		* \return Returns FALSE if there already is an option with this name
		* OR if a short option string (length == 1) is a digit. In that case no
		* action is peformed.
		*/
		bool defineOption(const std::string& _name,
			const std::string& _description = std::string(),
			OptionAttributes _attributes = NoOptionAttribute);
		/** Define an alternative name for an option that was previously defined by
		* defineOption().
		* \return Returns FALSE if there already is an option with the alternative
		* name or no option with the original name OR if a short option string
		* (length == 1) is a digit. In that case no action is performed.
		*/
		bool defineOptionAlternative(const std::string& _original,
			const std::string& _alternative);
		/** Returns whether _name is a defined option. */
		bool isDefinedOption(const std::string& _name) const;
		/** Returns whether _name is an option that was found while parsing
		* the command line arguments with the parse() method. In other word: This
		* method returns true if the string is an option AND it was given on the
		* parsed command line.
		*/
		bool foundOption(const std::string& _name) const;
		/** Define a help option. If this option is found a special error code is
		* returned by the parse method.
		* \attention If this method is called twice without an intermediate call
		* to the reset() method the previously set help option will remain a valid
		* option but is not detected as the special help option and will therefore
		* not cause the parse() method to return the special help error code.
		* \return Returns FALSE if there already is an option defined that equals
		* the short or long name.
		*/
		bool setHelpOption(const std::string& _longname = "h",
			const std::string& _shortname = "help",
			const std::string& _descr = "");
		/** Returns the number of read arguments. Arguments are efined as beeing
		* neither options nor option values and are specified at the end of the
		* command line after all options and their values. */
		unsigned int arguments() const;
		/** Returns the Nth argument. See arguments().
		* \return Argument string or an empty string if there was no argument of
		* that id.
		*/
		std::string argument(unsigned int _number) const;
		/** Get the complete argument vector. The order of the arguments in the
		* vector is the same as on the commandline.
		*/
		const std::vector<std::string>& allArguments() const;
		/** Add an error code and its description to the command line parser.
		* This will do nothing more than adding an entry to the usage description.
		*/
		void addErrorCode(int _code, const std::string& _descr = "");
		/** Set some string as a general description, that will be printed before
		* the list of available options.
		*/
		void setIntroductoryDescription(const std::string& _descr);
		/** Parse the command line arguments for all known options and arguments.
		* \return Error code with parsing result.
		* \retval NoParserError Everything went fine.
		* \retval ParserUnknownOption Unknown option was found.
		* \retval ParserMissingValue A value to a given option is missing.
		* \retval ParserOptionAfterArgument Option after an argument detected. All
		* options have to given before the first argument.
		* \retval ParserMalformedMultipleShortOption Malformed short option string.
		* \retval ParserRequiredOptionMissing Required option is missing.
		* \retval ParserHelpRequested Help option detected.
		*/
		ParserResults parse(int _argc, char** _argv);
		/** Return the value of an option.
		* \return Value of a commandline options given by the name of the option or
		* an empty string if there was no such option or the option required no
		* value.
		*/
		std::string optionValue(const std::string& _option) const;
		/** Reset the parser. Call this function if you want to parse another set of
		* command line arguments with the same parser object.
		*/
		void reset();
		/** Returns the name of the option that was responsible for a parser error.
		  * An empty string is returned if no error occured at all.
		  */
		const std::string& errorOption() const;
		/** This method can be used to evaluate parser error codes and generate a
		* human-readable description. In case of a help request error code the
		* usage description as returned by usageDescription() is printed.
		*/
		std::string parseErrorDescription(ParserResults _error_code) const;
		/** Returns a string with the usage descriptions for all options. The
		 * description string is formated to fit into a terminal of width _width.*/

		 /** Returns the key of a defined option with name _name or -1 if such option
		  * is not defined. */
		int optionKey(const std::string& _name) const;
	private:
		/** Returns a list of option names that are all alternative names associated
		 * with a single key value.
		 */
		std::list<std::string> getAllOptionAlternatives(unsigned int _key) const;

		/** The current maximum key value for an option. */
		unsigned int max_key;
		/** Map option names to a numeric key. */
		String2KeyMap option2key;

		/** Map option key to option attributes. */
		Key2AttributeMap option2attribute;

		/** Map option key to option description. */
		Key2StringMap option2descr;

		/** Map option key to option value. */
		Key2StringMap option2value;

		/** Map error code to its description. */
		std::map<int, std::string> errorcode2descr;

		/** Vector of command line arguments. */
		ArgumentContainer argument_container;

		/** General description to be returned as first part of the generated help page. */
		std::string intro_description;

		/** Holds the key for the help option. */
		unsigned int help_option;

		/** Holds the name of the option that was responsible for a parser error.
		*/
		std::string error_option;
	}; // class ArgvParser


	// Auxillary functions

	/** Returns whether the given string is a valid (correct syntax) option string.
	 * It has to fullfill the following criteria:
	 *  1. minimum length is 2 characters
	 *  2. Start with '-'
	 *  3. if if minimal length -> must not be '--'
	 *  4. first short option character must not be a digit (to distinguish negative numbers)
	 */
	bool isValidOptionString(const std::string& _string);

	/** Returns whether the given string is a valid (correct syntax) long option string.
	 * It has to fullfill the following criteria:
	 *  1. minimum length is 4 characters
	 *  2. Start with '--'
	 */
	bool isValidLongOptionString(const std::string& _string);

	/** Splits option and value string if they are given in the form 'option=value'.
	* \return Returns TRUE if a value was found.
	*/
	bool splitOptionAndValue(const std::string& _string, std::string& _option,
		std::string& _value);

	/** String tokenizer using standard C++ functions. Taken from here:
	 * http://gcc.gnu.org/onlinedocs/libstdc++/21_strings/howto.html#3
	 * Splits the string _in by _delimiters and store the tokens in _container.
	 */
	template <typename Container>
	void splitString(Container& _container, const std::string& _in,
		const char* const _delimiters = " \t\n")
	{
		const std::string::size_type len = _in.length();
		std::string::size_type i = 0;

		while (i < len)
		{
			// eat leading whitespace
			i = _in.find_first_not_of(_delimiters, i);
			if (i == std::string::npos)
				return;   // nothing left but white space

			// find the end of the token
			std::string::size_type j = _in.find_first_of(_delimiters, i);

			// push token
			if (j == std::string::npos)
			{
				_container.push_back(_in.substr(i));
				return;
			}
			else
				_container.push_back(_in.substr(i, j - i));

			// set up for next loop
			i = j + 1;
		}
	}

	/** Returns true if the character is a digit (what else?). */
	bool isDigit(const char& _char);

	/** Build a vector of integers from a string of the form:
	* '1,3-5,14,25-20'. This string will be expanded to a list of positive
	* integers with the following elements: 1,3,4,5,14,25,24,23,22,21,20.
	* All of the expanded elements will be added to the provided list.
	* \return Returns FALSE if there was any syntax error in the given string
	* In that case the function stops at the point where the error occured.
	* Only elements processed up to that point will be added to the expanded
	* list.
	* \attention This function can only handle unsigned integers!
	*/
	bool expandRangeStringToUInt(const std::string& _string,
		std::vector<unsigned int>& _expanded);
	/** Returns a copy of _str with whitespace removed from front and back. */
	std::string trimmedString(const std::string& _str);

	/** Formats a string of an arbitrary length to fit a terminal of width
	* _width and to be indented by _indent columns.
	*/
	std::string formatString(const std::string& _string,
		unsigned int _width,
		unsigned int _indent = 0);

};


using namespace std;

using namespace CommandLineProcessing;

class OptionParser {
public:
	OptionParser();

	static void setup();
	static void addOption(const string& option, const string& desc);
	static void parseOptions(int argc, char** argv);
	static string optionValue(const string& option);
	static bool foundOption(const string& option);

	void add(const string& option, const string& desc);
	int parse(string optionsText);
	string value(const string& option);
	bool has(const string& option);

	static float parsefloat(const string& option);
	static int parseInt(const string& option);
private:
	static ArgvParser cmd;
	ArgvParser myCmd;
};


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}	


__device__ float* out_g;
//GPU
__device__ void  partitionedStep(float* Y_new_, float* pars, float* algs, float* rhs, float* Y_old_, float t, float dt, float** strut = NULL);
__device__ void step(float* Y_new_, int n, float* as, float* bs, float* Y_old_, float dt);
__device__ void step(float* Y_new_, float* pars, float* algs, float* rhs, float* Y_old_, float t, float dt, float** strut);
__device__ void calc_algs_hh(float* algs, float* pars, float* Y_old_, float time)
{
	calc_xr1_inf = (1.0e+00 / (1.0e+00 + exp((((-2.60e+01) - V_old_) / 7.0e+00))));	//11+
	calc_alpha_xr1 = (4.50e+02 / (1.0e+00 + exp((((-4.50e+01) - V_old_) / 1.0e+01))));	//12
	calc_beta_xr1 = (6.0e+00 / (1.0e+00 + exp(((V_old_ + 3.0e+01) / 1.150e+01))));	//13
	calc_xr2_inf = (1.0e+00 / (1.0e+00 + exp(((V_old_ + 8.80e+01) / 2.40e+01))));	//16
	calc_alpha_xr2 = (3.0e+00 / (1.0e+00 + exp((((-6.0e+01) - V_old_) / 2.0e+01))));	//17
	calc_beta_xr2 = (1.120e+00 / (1.0e+00 + exp(((V_old_ - 6.0e+01) / 2.0e+01))));	//18
	calc_xs_inf = (1.0e+00 / (1.0e+00 + exp((((-5.0e+00) - V_old_) / 1.40e+01))));	//22
	calc_alpha_xs = (1.10e+03 / pow((1.0e+00 + exp((((-1.0e+01) - V_old_) / 6.0e+00))), 1.0 / 2.0));	//23
	calc_beta_xs = (1.0e+00 / (1.0e+00 + exp(((V_old_ - 6.0e+01) / 2.0e+01))));	//24

	calc_m_inf = (1.0e+00 / pow((1.0e+00 + exp((((-5.6860e+01) - V_old_) / 9.03e+00))), 2.0e+00));	//28
	calc_alpha_m = (1.0e+00 / (1.0e+00 + exp((((-6.0e+01) - V_old_) / 5.0e+00))));	//29
	calc_beta_m = ((1.0e-01 / (1.0e+00 + exp(((V_old_ + 3.50e+01) / 5.0e+00)))) + (1.0e-01 / (1.0e+00 + exp(((V_old_ - 5.0e+01) / 2.0e+02)))));	//30
	calc_h_inf = (1.0e+00 / pow((1.0e+00 + exp(((V_old_ + 7.1550e+01) / 7.430e+00))), 2.0e+00));	//33
	calc_alpha_h = (V_old_ < -40) ? 5.70e-02 * exp(((-(V_old_ + 8.0e+01)) / 6.80e+00)) : 0;	//34
	calc_beta_h = (V_old_ < -40) ? (2.70e+00 * exp((7.90e-02 * V_old_))) + (3.10e+05 * exp((3.4850e-01 * V_old_))) : 7.70e-01 / (1.30e-01 * (1.0 + exp(((V_old_ + 1.0660e+01) / (-1.110e+01)))));	//35
	calc_j_inf = (1.0e+00 / pow((1.0e+00 + exp(((V_old_ + 7.1550e+01) / 7.430e+00))), 2.0e+00));	//38
	calc_alpha_j = (V_old_ < -40) ? ((-2.5428e4) * exp(0.2444 * V_old_) - (6.948e-6) * exp(-0.04391 * V_old_)) * (V_old_ + 37.78) / (1. + exp(0.311 * (V_old_ + 79.23))) : 0;	//39
	calc_beta_j = (V_old_ < -40) ? 0.02424 * exp(-0.01052 * V_old_) / (1. + exp(-0.1378 * (V_old_ + 40.14))) : 0.6 * exp((0.057) * V_old_) / (1. + exp(-0.1 * (V_old_ + 32.)));	//40

	calc_d_inf = (1.0e+00 / (1.0e+00 + exp((((-5.0e+00) - V_old_) / 7.50e+00))));	//45
	calc_alpha_d = ((1.40e+00 / (1.0e+00 + exp((((-3.50e+01) - V_old_) / 1.30e+01)))) + 2.50e-01);	//46
	calc_beta_d = (1.40e+00 / (1.0e+00 + exp(((V_old_ + 5.0e+00) / 5.0e+00))));	//47
	calc_gamma_d = (1.0e+00 / (1.0e+00 + exp(((5.0e+01 - V_old_) / 2.0e+01))));	//48

	calc_f_inf = 1. / (1. + exp((V_old_ + 20) / 7));	//51 
	calc_tau_f = 1125 * exp(-(V_old_ + 27) * (V_old_ + 27) / 240) + 80 + 165 / (1. + exp((25 - V_old_) / 10));	//52 300 -> 240 ?
	//calc_tau_f = ( 3.0e+01+3.5e+02/(1.0e+00+exp((V_on_f+2.5e+01)/9.50e+00)) );

	calc_tau_fCa = 2.0;	//58
	calc_s_inf = (1.0e+00 / (1.0e+00 + exp(((V_old_ + 2.0e+01) / 5.0e+00))));	//63
	calc_tau_s = ((8.50e+01 * exp(((-pow((V_old_ + 4.50e+01), 2.0e+00)) / 3.20e+02))) + (5.0e+00 / (1.0e+00 + exp(((V_old_ - 2.0e+01) / 5.0e+00)))) + 3.0e+00);	//64
	calc_r_inf = (1.0e+00 / (1.0e+00 + exp(((2.0e+01 - V_old_) / 6.0e+00))));	//66
	calc_tau_r = ((9.50e+00 * exp(((-pow((V_old_ + 4.0e+01), 2.0e+00)) / 1.80e+03))) + 8.0e-01);	//67
	calc_g_inf = Ca_i_old_ < 3.50e-04 ? 1.0e+00 / (1.0e+00 + pow((Ca_i_old_ / 3.50e-04), 6.0e+00)) : 1.0e+00 / (1.0e+00 + pow((Ca_i_old_ / 3.50e-04), 1.60e+01));	//76
	calc_tau_xr1 = (1.0e+00 * calc_alpha_xr1 * calc_beta_xr1);	//14
	calc_tau_xr2 = (1.0e+00 * calc_alpha_xr2 * calc_beta_xr2);	//19
	calc_tau_xs = (1.0e+00 * calc_alpha_xs * calc_beta_xs);	//25
	calc_tau_m = (1.0e+00 * calc_alpha_m * calc_beta_m);	//31
	calc_tau_h = (1.0e+00 / (calc_alpha_h + calc_beta_h));	//36
	calc_tau_j = (1.0e+00 / (calc_alpha_j + calc_beta_j));	//41
	calc_tau_d = ((1.0e+00 * calc_alpha_d * calc_beta_d) + calc_gamma_d);	//49

	calc_alpha_fCa = (1.0e+00 / (1.0e+00 + pow((Ca_i_old_ / 3.250e-04), 8.0e+00)));	//54
	calc_beta_fCa = (1.0e-01 / (1.0e+00 + exp(((Ca_i_old_ - 5.0e-04) / 1.0e-04))));	//55
	calc_gama_fCa = (2.0e-01 / (1.0e+00 + exp(((Ca_i_old_ - 7.50e-04) / 8.0e-04))));	//56
	calc_fCa_inf = ((calc_alpha_fCa + calc_beta_fCa + calc_gama_fCa + 2.30e-01) / 1.460e+00);	//57

	calc_d_g = ((calc_g_inf - g_old_) / tau_g);	//77
	calc_d_fCa = ((calc_fCa_inf - fCa_old_) / calc_tau_fCa);	//59
}
__device__ void calc_hh_coeff(float* a, float* b, float* pars, float* algs, float* Y_old_, float t)
{
	calc_algs_hh(algs, pars, Y_old_, t);

	Xr1_a_ = -1.0 / calc_tau_xr1;	// 15
	Xr2_a_ = -1.0 / calc_tau_xr2;	// 20
	Xs_a_ = -1.0 / calc_tau_xs;	// 26
	m_a_ = -1.0 / calc_tau_m;	// 32
	h_a_ = -1.0 / calc_tau_h;	// 37
	j_a_ = -1.0 / calc_tau_j;	// 42
	d_a_ = -1.0 / calc_tau_d;	// 50
	f_a_ = -1.0 / calc_tau_f;	// 53
	//fCa_a_= (calc_fCa_inf>fCa_old_&& V_old_>-6.0e+01) ? 0.0 : -1.0/calc_tau_fCa;	// 56
	fCa_a_ = -1.0 / calc_tau_fCa;
	s_a_ = -1.0 / calc_tau_s;	// 64
	r_a_ = -1.0 / calc_tau_r;	// 67
	//g_a_=(calc_g_inf>g_old_&&V_old_>-6.0e+01) ? 0.0 : -1.0/tau_g;
	g_a_ = -1.0 / tau_g;

	Xr1_b_ = (((calc_xr1_inf) / calc_tau_xr1));
	Xr2_b_ = (((calc_xr2_inf) / calc_tau_xr2));
	Xs_b_ = (((calc_xs_inf) / calc_tau_xs));
	m_b_ = (((calc_m_inf) / calc_tau_m));
	h_b_ = (((calc_h_inf) / calc_tau_h));
	j_b_ = (((calc_j_inf) / calc_tau_j));
	d_b_ = (((calc_d_inf) / calc_tau_d));
	f_b_ = (((calc_f_inf) / calc_tau_f));
	//fCa_b_= (calc_fCa_inf>fCa_old_&& V_old_>-6.0e+01) ? 0.0 : calc_fCa_inf/calc_tau_fCa;
	fCa_b_ = calc_fCa_inf / calc_tau_fCa;
	s_b_ = (((calc_s_inf) / calc_tau_s));
	r_b_ = (((calc_r_inf) / calc_tau_r));
	//g_b_ = (calc_g_inf>g_old_&&V_old_>-6.0e+01) ? 0.0 : calc_g_inf/tau_g;
	g_b_ = calc_g_inf / tau_g;

	if ((fCa_old_ * fCa_a_ + fCa_b_) > 0.0 && V_old_ > -37) {
		fCa_a_ = fCa_b_ = 0.0;	// 56
	}
	if ((g_old_ * g_a_ + g_b_) > 0.0 && V_old_ > -37) {
		g_a_ = g_b_ = 0.0;	// 56
	}
}
__device__ void calc_rhs_mk(float* rhs, float* pars, float* algs, float* Y_old_, float t)
{
}
__device__ float calc_stimulus(float* args, float t)
{
	if (stim_state < 0)
		return 0;
	if (stim_state > 0)
		return stim_amplitude;

	float t_since_last_tick = t - floor(t / stim_period) * stim_period;
	float pulse_end = stim_start + stim_duration;
	if (t_since_last_tick >= stim_start && t_since_last_tick <= pulse_end) {
		return stim_amplitude;
	}
	else return 0;
}
__device__ void calc_algs_nl(float* algs, float* args, float* Y_old_, float time,int tid,int N)
{
	
	calc_i_Stim = calc_stimulus(args, time);	//0
	calc_E_Na = (((R * T) / F) * log((Na_o / Na_i_old_)));	//2
	calc_E_K = (((R * T) / F) * log((K_o / K_i_old_)));	//3
	calc_E_Ks = (((R * T) / F) * log(((K_o + (P_kna * Na_o)) / (K_i_old_ + (P_kna * Na_i_old_)))));	//4
	calc_E_Ca = (((5.0e-01 * R * T) / F) * log((Ca_o / Ca_i_old_)));	//5
	calc_i_CaL = ((((g_CaL * d_old_ * f_old_ * fCa_old_ * 4.0e+00 * V_old_ * pow(F, 2.0e+00)) / (R * T)) * ((Ca_i_old_ * exp(((2.0e+00 * V_old_ * F) / (R * T)))) - (3.410e-01 * Ca_o))) / (exp(((2.0e+00 * V_old_ * F) / (R * T))) - 1.0e+00));	//44
	calc_i_NaK = (((((P_NaK * K_o) / (K_o + K_mk)) * Na_i_old_) / (Na_i_old_ + K_mNa)) / (1.0e+00 + (1.2450e-01 * exp((((-1.0e-01) * V_old_ * F) / (R * T)))) + (3.530e-02 * exp((((-V_old_) * F) / (R * T))))));	//69
	calc_i_NaCa = ((K_NaCa * ((exp(((gamma * V_old_ * F) / (R * T))) * pow(Na_i_old_, 3.0e+00) * Ca_o) - (exp((((gamma - 1.0e+00) * V_old_ * F) / (R * T))) * pow(Na_o, 3.0e+00) * Ca_i_old_ * alpha))) / ((pow(Km_Nai, 3.0e+00) + pow(Na_o, 3.0e+00)) * (Km_Ca + Ca_o) * (1.0e+00 + (K_sat * exp((((gamma - 1.0e+00) * V_old_ * F) / (R * T)))))));	//70
	calc_i_p_Ca = ((g_pCa * Ca_i_old_) / (Ca_i_old_ + K_pCa));	//71
	calc_i_rel = ((((a_rel * pow(Ca_SR_old_, 2.0e+00)) / (pow(b_rel, 2.0e+00) + pow(Ca_SR_old_, 2.0e+00))) + c_rel) * d_old_ * g_old_);	//73
	calc_i_up = (Vmax_up / (1.0e+00 + (pow(K_up, 2.0e+00) / pow(Ca_i_old_, 2.0e+00))));	//74
	calc_i_leak = (V_leak * (Ca_SR_old_ - Ca_i_old_));	//75
	calc_Ca_i_bufc = (1.0e+00 / (1.0e+00 + ((Buf_c * K_buf_c) / pow((Ca_i_old_ + K_buf_c), 2.0e+00))));	//79
	calc_Ca_sr_bufsr = (1.0e+00 / (1.0e+00 + ((Buf_sr * K_buf_sr) / pow((Ca_SR_old_ + K_buf_sr), 2.0e+00))));	//80
	calc_i_Kr = (g_Kr * pow((K_o / 5.40e+00), 1.0 / 2.0) * Xr1_old_ * Xr2_old_ * (V_old_ - calc_E_K));	//10
	calc_i_Ks = (g_Ks * pow(Xs_old_, 2.0e+00) * (V_old_ - calc_E_Ks));	//21
	calc_i_Na = (g_Na * pow(m_old_, 3.0e+00) * h_old_ * j_old_ * (V_old_ - calc_E_Na));	//27
	calc_i_b_Na = (g_bna * (V_old_ - calc_E_Na));	//43
	calc_i_b_Ca = (g_bca * (V_old_ - calc_E_Ca));	//61
	calc_i_to = (g_to * r_old_ * s_old_ * (V_old_ - calc_E_K));	//62
	calc_i_p_K = ((g_pK * (V_old_ - calc_E_K)) / (1.0e+00 + exp(((2.50e+01 - V_old_) / 5.980e+00))));	//72
	calc_alpha_K1 = (1.0e-01 / (1.0e+00 + exp((6.0e-02 * ((V_old_ - calc_E_K) - 2.0e+02)))));	//6
	calc_beta_K1 = (((3.0e+00 * exp((2.0e-04 * ((V_old_ - calc_E_K) + 1.0e+02)))) + (1.0e+00 * exp((1.0e-01 * ((V_old_ - calc_E_K) - 1.0e+01))))) / (1.0e+00 + exp(((-5.0e-01) * (V_old_ - calc_E_K)))));	//7
	calc_xK1_inf = (calc_alpha_K1 / (calc_alpha_K1 + calc_beta_K1));	//8
	calc_i_K1 = (g_K1 * calc_xK1_inf * pow((K_o / 5.40e+00), 1.0 / 2.0) * (V_old_ - calc_E_K));	//9
}
__device__ void calc_rhs_nl(float* rhs, float* args, float* algs, float* Y_old_, float t,int tid, int N)
{


	calc_algs_nl(algs, args, Y_old_, t,tid,N);
	float gkatp_f = 4E6 / (1 + pow((atp / 0.25), 2.0)) * (195E-6 / (5E+3)) * pow((K_o / K_i_old_), 0.24);

	//printf("\n%.15f\n", gkatp_f * (V_old_ - 5E3));

	ikatp_f = gkatp_f * (V_old_ - 5E3);
	V_f_ = -(-ikatp_f + calc_i_K1 + +calc_i_Kr + calc_i_Ks + calc_i_CaL + calc_i_NaK + calc_i_Na + calc_i_b_Na + calc_i_NaCa + calc_i_b_Ca + calc_i_p_K + calc_i_p_Ca + calc_i_Stim);
	Ca_i_f_ = ((calc_Ca_i_bufc * (((calc_i_leak - calc_i_up) + calc_i_rel) - (((1.0e+00 * ((calc_i_CaL + calc_i_b_Ca + calc_i_p_Ca) - (2.0e+00 * calc_i_NaCa))) / (2.0e+00 * 1.0e+00 * V_c * F)) * Cm))));	// 81
	Ca_SR_f_ = ((((calc_Ca_sr_bufsr * V_c) / V_sr) * (calc_i_up - (calc_i_rel + calc_i_leak))));	// 82
	Na_i_f_ = ((((-1.0e+00) * (calc_i_Na + calc_i_b_Na + (3.0e+00 * calc_i_NaK) + (3.0e+00 * calc_i_NaCa)) * Cm) / (1.0e+00 * V_c * F)));	// 83
	K_i_f_ = ((((-1.0e+00) * ((calc_i_K1 + calc_i_to + calc_i_Kr + calc_i_Ks + calc_i_p_K + calc_i_Stim) - (2.0e+00 * calc_i_NaK)) * Cm) / (1.0e+00 * V_c * F)));	// 84

}
__device__ void initModel(float* pars, float* Y_old_, float* args)
{

	unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
	unsigned int tid = threadIdx.x + (blockDim.x) * bid;;

	unsigned int N = blockDim.x * gridDim.x * gridDim.y;
	



	printf("\n ss %d %f %f %f %f\n",tid, g_Na,atp,K_o,g_CaL);

	// K_i_old_ = args[0];


	


	V_old_ = -8.620e+01;

	Xr1_old_ = 0.0e+00;

	Xr2_old_ = 1.0e+00;

	Xs_old_ = 0.0e+00;
	m_old_ = 0.0e+00;
	h_old_ = 7.50e-01;
	j_old_ = 7.50e-01;
	d_old_ = 0.0e+00;
	f_old_ = 1.0e+00;
	fCa_old_ = 1.0e+00;
	s_old_ = 1.0e+00;
	r_old_ = 0.0e+00;

	g_old_ = 1.0e+00;
	Ca_i_old_ = 0.00008;//2.0e-04;
	Ca_SR_old_ = 0.56;//2.0e-01;
	Na_i_old_ = 11.6;//1.160e+01;

	K_i_old_ = 138.3;//1.3830e+02;


}
__device__ void step(float* Y_new_, float* pars, float* algs, float* rhs, float* Y_old_, float t, float dt, float** strut, int tid, int N)
{


	partitionedStep(Y_new_, pars, algs, rhs, Y_old_, t, dt);
	calc_rhs_mk(rhs, pars, algs, Y_old_, t);

	for (int l = MKStart; l < MKEnd; l++)
	{
		Y_new_[l] = Y_old_[l] + dt * rhs[l];


	}

	calc_rhs_nl(rhs, pars, algs, Y_old_, t,tid,N);
	for (int l = NLStart; l < NLEnd; l++)
		Y_new_[l] = Y_old_[l] + dt * rhs[l];
}
__device__ void partitionedStep(float* Y_new_, float* pars, float* algs, float* rhs, float* Y_old_, float t, float dt, float** strut)
{

	float* as = &(rhs[HHStart]);
	float* bs = &(rhs[nStates]);

	calc_hh_coeff(as, bs, pars, algs, Y_old_, t);

	step(&(Y_new_[HHStart]), nStates_HH, as, bs, &(Y_old_[HHStart]), dt);
}
__device__ void step(float* Y_new_, int n, float* as, float* bs, float* Y_old_, float dt)
{


	for (int i = 0; i < n; i++) {
		if (abs(as[i]) < EPSILON) { // TODO change to epsilon comparison
			Y_new_[i] = Y_old_[i] + dt * (Y_old_[i] * as[i] + bs[i]);
		}
		else {
			float aux = bs[i] / as[i];
			Y_new_[i] = exp(as[i] * dt) * (Y_old_[i] + aux) - aux;
		}
	}

}
__global__ void solveFixed( float* out_g,float dt, float dt_save, float tf, float* args);



//CPU
void  partitionedStepC(float* Y_new_, float* pars, float* algs, float* rhs, float* Y_old_, float t, float dt, float** strut = NULL);
void stepC(float* Y_new_, int n, float* as, float* bs, float* Y_old_, float dt);
void stepC(float* Y_new_, float* pars, float* algs, float* rhs, float* Y_old_, float t, float dt, float** strut);
void calc_algs_hhC(float* algs, float* pars, float* Y_old_, float time)
{
	calc_xr1_inf = (1.0e+00 / (1.0e+00 + exp((((-2.60e+01) - V_old_) / 7.0e+00))));	//11+
	calc_alpha_xr1 = (4.50e+02 / (1.0e+00 + exp((((-4.50e+01) - V_old_) / 1.0e+01))));	//12
	calc_beta_xr1 = (6.0e+00 / (1.0e+00 + exp(((V_old_ + 3.0e+01) / 1.150e+01))));	//13
	calc_xr2_inf = (1.0e+00 / (1.0e+00 + exp(((V_old_ + 8.80e+01) / 2.40e+01))));	//16
	calc_alpha_xr2 = (3.0e+00 / (1.0e+00 + exp((((-6.0e+01) - V_old_) / 2.0e+01))));	//17
	calc_beta_xr2 = (1.120e+00 / (1.0e+00 + exp(((V_old_ - 6.0e+01) / 2.0e+01))));	//18
	calc_xs_inf = (1.0e+00 / (1.0e+00 + exp((((-5.0e+00) - V_old_) / 1.40e+01))));	//22
	calc_alpha_xs = (1.10e+03 / pow((1.0e+00 + exp((((-1.0e+01) - V_old_) / 6.0e+00))), 1.0 / 2.0));	//23
	calc_beta_xs = (1.0e+00 / (1.0e+00 + exp(((V_old_ - 6.0e+01) / 2.0e+01))));	//24

	calc_m_inf = (1.0e+00 / pow((1.0e+00 + exp((((-5.6860e+01) - V_old_) / 9.03e+00))), 2.0e+00));	//28
	calc_alpha_m = (1.0e+00 / (1.0e+00 + exp((((-6.0e+01) - V_old_) / 5.0e+00))));	//29
	calc_beta_m = ((1.0e-01 / (1.0e+00 + exp(((V_old_ + 3.50e+01) / 5.0e+00)))) + (1.0e-01 / (1.0e+00 + exp(((V_old_ - 5.0e+01) / 2.0e+02)))));	//30
	calc_h_inf = (1.0e+00 / pow((1.0e+00 + exp(((V_old_ + 7.1550e+01) / 7.430e+00))), 2.0e+00));	//33
	calc_alpha_h = (V_old_ < -40) ? 5.70e-02 * exp(((-(V_old_ + 8.0e+01)) / 6.80e+00)) : 0;	//34
	calc_beta_h = (V_old_ < -40) ? (2.70e+00 * exp((7.90e-02 * V_old_))) + (3.10e+05 * exp((3.4850e-01 * V_old_))) : 7.70e-01 / (1.30e-01 * (1.0 + exp(((V_old_ + 1.0660e+01) / (-1.110e+01)))));	//35
	calc_j_inf = (1.0e+00 / pow((1.0e+00 + exp(((V_old_ + 7.1550e+01) / 7.430e+00))), 2.0e+00));	//38
	calc_alpha_j = (V_old_ < -40) ? ((-2.5428e4) * exp(0.2444 * V_old_) - (6.948e-6) * exp(-0.04391 * V_old_)) * (V_old_ + 37.78) / (1. + exp(0.311 * (V_old_ + 79.23))) : 0;	//39
	calc_beta_j = (V_old_ < -40) ? 0.02424 * exp(-0.01052 * V_old_) / (1. + exp(-0.1378 * (V_old_ + 40.14))) : 0.6 * exp((0.057) * V_old_) / (1. + exp(-0.1 * (V_old_ + 32.)));	//40

	calc_d_inf = (1.0e+00 / (1.0e+00 + exp((((-5.0e+00) - V_old_) / 7.50e+00))));	//45
	calc_alpha_d = ((1.40e+00 / (1.0e+00 + exp((((-3.50e+01) - V_old_) / 1.30e+01)))) + 2.50e-01);	//46
	calc_beta_d = (1.40e+00 / (1.0e+00 + exp(((V_old_ + 5.0e+00) / 5.0e+00))));	//47
	calc_gamma_d = (1.0e+00 / (1.0e+00 + exp(((5.0e+01 - V_old_) / 2.0e+01))));	//48

	calc_f_inf = 1. / (1. + exp((V_old_ + 20) / 7));	//51 
	calc_tau_f = 1125 * exp(-(V_old_ + 27) * (V_old_ + 27) / 240) + 80 + 165 / (1. + exp((25 - V_old_) / 10));	//52 300 -> 240 ?
	//calc_tau_f = ( 3.0e+01+3.5e+02/(1.0e+00+exp((V_on_f+2.5e+01)/9.50e+00)) );

	calc_tau_fCa = 2.0;	//58
	calc_s_inf = (1.0e+00 / (1.0e+00 + exp(((V_old_ + 2.0e+01) / 5.0e+00))));	//63
	calc_tau_s = ((8.50e+01 * exp(((-pow((V_old_ + 4.50e+01), 2.0e+00)) / 3.20e+02))) + (5.0e+00 / (1.0e+00 + exp(((V_old_ - 2.0e+01) / 5.0e+00)))) + 3.0e+00);	//64
	calc_r_inf = (1.0e+00 / (1.0e+00 + exp(((2.0e+01 - V_old_) / 6.0e+00))));	//66
	calc_tau_r = ((9.50e+00 * exp(((-pow((V_old_ + 4.0e+01), 2.0e+00)) / 1.80e+03))) + 8.0e-01);	//67
	calc_g_inf = Ca_i_old_ < 3.50e-04 ? 1.0e+00 / (1.0e+00 + pow((Ca_i_old_ / 3.50e-04), 6.0e+00)) : 1.0e+00 / (1.0e+00 + pow((Ca_i_old_ / 3.50e-04), 1.60e+01));	//76
	calc_tau_xr1 = (1.0e+00 * calc_alpha_xr1 * calc_beta_xr1);	//14
	calc_tau_xr2 = (1.0e+00 * calc_alpha_xr2 * calc_beta_xr2);	//19
	calc_tau_xs = (1.0e+00 * calc_alpha_xs * calc_beta_xs);	//25
	calc_tau_m = (1.0e+00 * calc_alpha_m * calc_beta_m);	//31
	calc_tau_h = (1.0e+00 / (calc_alpha_h + calc_beta_h));	//36
	calc_tau_j = (1.0e+00 / (calc_alpha_j + calc_beta_j));	//41
	calc_tau_d = ((1.0e+00 * calc_alpha_d * calc_beta_d) + calc_gamma_d);	//49

	calc_alpha_fCa = (1.0e+00 / (1.0e+00 + pow((Ca_i_old_ / 3.250e-04), 8.0e+00)));	//54
	calc_beta_fCa = (1.0e-01 / (1.0e+00 + exp(((Ca_i_old_ - 5.0e-04) / 1.0e-04))));	//55
	calc_gama_fCa = (2.0e-01 / (1.0e+00 + exp(((Ca_i_old_ - 7.50e-04) / 8.0e-04))));	//56
	calc_fCa_inf = ((calc_alpha_fCa + calc_beta_fCa + calc_gama_fCa + 2.30e-01) / 1.460e+00);	//57

	calc_d_g = ((calc_g_inf - g_old_) / tau_g);	//77
	calc_d_fCa = ((calc_fCa_inf - fCa_old_) / calc_tau_fCa);	//59
}
void calc_hh_coeffC(float* a, float* b, float* pars, float* algs, float* Y_old_, float t)
{
	calc_algs_hhC(algs, pars, Y_old_, t);

	Xr1_a_ = -1.0 / calc_tau_xr1;	// 15
	Xr2_a_ = -1.0 / calc_tau_xr2;	// 20
	Xs_a_ = -1.0 / calc_tau_xs;	// 26
	m_a_ = -1.0 / calc_tau_m;	// 32
	h_a_ = -1.0 / calc_tau_h;	// 37
	j_a_ = -1.0 / calc_tau_j;	// 42
	d_a_ = -1.0 / calc_tau_d;	// 50
	f_a_ = -1.0 / calc_tau_f;	// 53
	//fCa_a_= (calc_fCa_inf>fCa_old_&& V_old_>-6.0e+01) ? 0.0 : -1.0/calc_tau_fCa;	// 56
	fCa_a_ = -1.0 / calc_tau_fCa;
	s_a_ = -1.0 / calc_tau_s;	// 64
	r_a_ = -1.0 / calc_tau_r;	// 67
	//g_a_=(calc_g_inf>g_old_&&V_old_>-6.0e+01) ? 0.0 : -1.0/tau_g;
	g_a_ = -1.0 / tau_g;

	Xr1_b_ = (((calc_xr1_inf) / calc_tau_xr1));
	Xr2_b_ = (((calc_xr2_inf) / calc_tau_xr2));
	Xs_b_ = (((calc_xs_inf) / calc_tau_xs));
	m_b_ = (((calc_m_inf) / calc_tau_m));
	h_b_ = (((calc_h_inf) / calc_tau_h));
	j_b_ = (((calc_j_inf) / calc_tau_j));
	d_b_ = (((calc_d_inf) / calc_tau_d));
	f_b_ = (((calc_f_inf) / calc_tau_f));
	//fCa_b_= (calc_fCa_inf>fCa_old_&& V_old_>-6.0e+01) ? 0.0 : calc_fCa_inf/calc_tau_fCa;
	fCa_b_ = calc_fCa_inf / calc_tau_fCa;
	s_b_ = (((calc_s_inf) / calc_tau_s));
	r_b_ = (((calc_r_inf) / calc_tau_r));
	//g_b_ = (calc_g_inf>g_old_&&V_old_>-6.0e+01) ? 0.0 : calc_g_inf/tau_g;
	g_b_ = calc_g_inf / tau_g;

	if ((fCa_old_ * fCa_a_ + fCa_b_) > 0.0 && V_old_ > -37) {
		fCa_a_ = fCa_b_ = 0.0;	// 56
	}
	if ((g_old_ * g_a_ + g_b_) > 0.0 && V_old_ > -37) {
		g_a_ = g_b_ = 0.0;	// 56
	}
}
void calc_rhs_mkC(float* rhs, float* pars, float* algs, float* Y_old_, float t)
{
}
float calc_stimulusC(float* pars, float t)
{
	if (stim_state < 0)
		return 0;
	if (stim_state > 0)
		return stim_amplitude;

	float t_since_last_tick = t - floor(t / stim_period) * stim_period;
	float pulse_end = stim_start + stim_duration;
	if (t_since_last_tick >= stim_start && t_since_last_tick <= pulse_end) {
		return stim_amplitude;
	}
	else return 0;
}
void calc_algs_nlC(float* algs, float* args, float* Y_old_, float time,int tid, int N)
{
	calc_i_Stim = calc_stimulusC(args, time);	//0
	calc_E_Na = (((R * T) / F) * log((Na_o / Na_i_old_)));	//2
	calc_E_K = (((R * T) / F) * log((K_o / K_i_old_)));	//3
	calc_E_Ks = (((R * T) / F) * log(((K_o + (P_kna * Na_o)) / (K_i_old_ + (P_kna * Na_i_old_)))));	//4
	calc_E_Ca = (((5.0e-01 * R * T) / F) * log((Ca_o / Ca_i_old_)));	//5
	calc_i_CaL = ((((g_CaL * d_old_ * f_old_ * fCa_old_ * 4.0e+00 * V_old_ * pow(F, 2.0e+00)) / (R * T)) * ((Ca_i_old_ * exp(((2.0e+00 * V_old_ * F) / (R * T)))) - (3.410e-01 * Ca_o))) / (exp(((2.0e+00 * V_old_ * F) / (R * T))) - 1.0e+00));	//44
	calc_i_NaK = (((((P_NaK * K_o) / (K_o + K_mk)) * Na_i_old_) / (Na_i_old_ + K_mNa)) / (1.0e+00 + (1.2450e-01 * exp((((-1.0e-01) * V_old_ * F) / (R * T)))) + (3.530e-02 * exp((((-V_old_) * F) / (R * T))))));	//69
	calc_i_NaCa = ((K_NaCa * ((exp(((gamma * V_old_ * F) / (R * T))) * pow(Na_i_old_, 3.0e+00) * Ca_o) - (exp((((gamma - 1.0e+00) * V_old_ * F) / (R * T))) * pow(Na_o, 3.0e+00) * Ca_i_old_ * alpha))) / ((pow(Km_Nai, 3.0e+00) + pow(Na_o, 3.0e+00)) * (Km_Ca + Ca_o) * (1.0e+00 + (K_sat * exp((((gamma - 1.0e+00) * V_old_ * F) / (R * T)))))));	//70
	calc_i_p_Ca = ((g_pCa * Ca_i_old_) / (Ca_i_old_ + K_pCa));	//71
	calc_i_rel = ((((a_rel * pow(Ca_SR_old_, 2.0e+00)) / (pow(b_rel, 2.0e+00) + pow(Ca_SR_old_, 2.0e+00))) + c_rel) * d_old_ * g_old_);	//73
	calc_i_up = (Vmax_up / (1.0e+00 + (pow(K_up, 2.0e+00) / pow(Ca_i_old_, 2.0e+00))));	//74
	calc_i_leak = (V_leak * (Ca_SR_old_ - Ca_i_old_));	//75
	calc_Ca_i_bufc = (1.0e+00 / (1.0e+00 + ((Buf_c * K_buf_c) / pow((Ca_i_old_ + K_buf_c), 2.0e+00))));	//79
	calc_Ca_sr_bufsr = (1.0e+00 / (1.0e+00 + ((Buf_sr * K_buf_sr) / pow((Ca_SR_old_ + K_buf_sr), 2.0e+00))));	//80
	calc_i_Kr = (g_Kr * pow((K_o / 5.40e+00), 1.0 / 2.0) * Xr1_old_ * Xr2_old_ * (V_old_ - calc_E_K));	//10
	calc_i_Ks = (g_Ks * pow(Xs_old_, 2.0e+00) * (V_old_ - calc_E_Ks));	//21
	calc_i_Na = (g_Na * pow(m_old_, 3.0e+00) * h_old_ * j_old_ * (V_old_ - calc_E_Na));	//27
	calc_i_b_Na = (g_bna * (V_old_ - calc_E_Na));	//43
	calc_i_b_Ca = (g_bca * (V_old_ - calc_E_Ca));	//61
	calc_i_to = (g_to * r_old_ * s_old_ * (V_old_ - calc_E_K));	//62
	calc_i_p_K = ((g_pK * (V_old_ - calc_E_K)) / (1.0e+00 + exp(((2.50e+01 - V_old_) / 5.980e+00))));	//72
	calc_alpha_K1 = (1.0e-01 / (1.0e+00 + exp((6.0e-02 * ((V_old_ - calc_E_K) - 2.0e+02)))));	//6
	calc_beta_K1 = (((3.0e+00 * exp((2.0e-04 * ((V_old_ - calc_E_K) + 1.0e+02)))) + (1.0e+00 * exp((1.0e-01 * ((V_old_ - calc_E_K) - 1.0e+01))))) / (1.0e+00 + exp(((-5.0e-01) * (V_old_ - calc_E_K)))));	//7
	calc_xK1_inf = (calc_alpha_K1 / (calc_alpha_K1 + calc_beta_K1));	//8
	calc_i_K1 = (g_K1 * calc_xK1_inf * pow((K_o / 5.40e+00), 1.0 / 2.0) * (V_old_ - calc_E_K));	//9
}
void calc_rhs_nlC(float* rhs, float* args, float* algs, float* Y_old_, float t,int tid,int  N)
{


	calc_algs_nlC(algs, args, Y_old_, t,tid,N);
	float gkatp_f = 4E6 / (1 + pow((atp / 0.25), 2.0)) * (195E-6 / (5E+3)) * pow((K_o / K_i_old_), 0.24);

	
	ikatp_f = gkatp_f * (V_old_ - 5E3);
	V_f_ = -(-ikatp_f + calc_i_K1 + +calc_i_Kr + calc_i_Ks + calc_i_CaL + calc_i_NaK + calc_i_Na + calc_i_b_Na + calc_i_NaCa + calc_i_b_Ca + calc_i_p_K + calc_i_p_Ca + calc_i_Stim);
	Ca_i_f_ = ((calc_Ca_i_bufc * (((calc_i_leak - calc_i_up) + calc_i_rel) - (((1.0e+00 * ((calc_i_CaL + calc_i_b_Ca + calc_i_p_Ca) - (2.0e+00 * calc_i_NaCa))) / (2.0e+00 * 1.0e+00 * V_c * F)) * Cm))));	// 81
	Ca_SR_f_ = ((((calc_Ca_sr_bufsr * V_c) / V_sr) * (calc_i_up - (calc_i_rel + calc_i_leak))));	// 82
	Na_i_f_ = ((((-1.0e+00) * (calc_i_Na + calc_i_b_Na + (3.0e+00 * calc_i_NaK) + (3.0e+00 * calc_i_NaCa)) * Cm) / (1.0e+00 * V_c * F)));	// 83
	K_i_f_ = ((((-1.0e+00) * ((calc_i_K1 + calc_i_to + calc_i_Kr + calc_i_Ks + calc_i_p_K + calc_i_Stim) - (2.0e+00 * calc_i_NaK)) * Cm) / (1.0e+00 * V_c * F)));	// 84

}
void initModelC(float* pars, float* Y_old_, float* args, int tid, int N)
{




	
	printf("\n ss %d %f %f %f %f\n", tid, g_Na, atp, K_o, g_CaL);

	V_old_ = -8.620e+01;

	Xr1_old_ = 0.0e+00;

	Xr2_old_ = 1.0e+00;

	Xs_old_ = 0.0e+00;
	m_old_ = 0.0e+00;
	h_old_ = 7.50e-01;
	j_old_ = 7.50e-01;
	d_old_ = 0.0e+00;
	f_old_ = 1.0e+00;
	fCa_old_ = 1.0e+00;
	s_old_ = 1.0e+00;
	r_old_ = 0.0e+00;

	g_old_ = 1.0e+00;
	Ca_i_old_ = 0.00008;//2.0e-04;
	Ca_SR_old_ = 0.56;//2.0e-01;
	Na_i_old_ = 11.6;//1.160e+01;

	K_i_old_ = 138.3;//1.3830e+02;


}
void stepC(float* Y_new_, float* pars, float* algs, float* rhs, float* Y_old_, float t, float dt, float** strut,int tid, int N)
{


	partitionedStepC(Y_new_, pars, algs, rhs, Y_old_, t, dt);
	calc_rhs_mkC(rhs, pars, algs, Y_old_, t);

	for (int l = MKStart; l < MKEnd; l++)
	{
		Y_new_[l] = Y_old_[l] + dt * rhs[l];


	}

	calc_rhs_nlC(rhs, pars, algs, Y_old_, t, tid, N);
	for (int l = NLStart; l < NLEnd; l++)
		Y_new_[l] = Y_old_[l] + dt * rhs[l];
}
void partitionedStepC(float* Y_new_, float* pars, float* algs, float* rhs, float* Y_old_, float t, float dt, float** strut)
{
	float* as = &(rhs[HHStart]);
	float* bs = &(rhs[nStates]);

	calc_hh_coeffC(as, bs, pars, algs, Y_old_, t);

	stepC(&(Y_new_[HHStart]), nStates_HH, as, bs, &(Y_old_[HHStart]), dt);
}
void stepC(float* Y_new_, int n, float* as, float* bs, float* Y_old_, float dt)
{


	for (int i = 0; i < n; i++) {
		if (abs(as[i]) < EPSILON) { // TODO change to epsilon comparison
			Y_new_[i] = Y_old_[i] + dt * (Y_old_[i] * as[i] + bs[i]);
		}
		else {
			float aux = bs[i] / as[i];
			Y_new_[i] = exp(as[i] * dt) * (Y_old_[i] + aux) - aux;
		}
	}

}
void solveFixedCpu(float* out, float dt, float dt_save, float tf, float* args, int tid, int N);


int main(int argc, char** argv)
{

	

	//Simulation Parameters
	OptionParser::addOption("model", "Model: 0 -> ten Tusscher 2004, 1 -> Fox 2002, 2 -> Bondarenko 2004");
	OptionParser::addOption("method", "Method: 0 -> Euler, 1 -> Rush Larsen, 2 -> Euler ADP, 3 -> Rush Larsen ADP, 4 -> UNI");
	OptionParser::addOption("dt", "Base time step.");
	OptionParser::addOption("dt_save", "Time step for saving.");
	OptionParser::addOption("tf", "Final time");
	OptionParser::addOption("ti", "Start output time");
	OptionParser::addOption("dt_max", "Maximum time step for adaptive solvers.");
	OptionParser::addOption("rel_tol", "Relative tolerance for adaptive solvers.");
	OptionParser::addOption("outputFile", "Filename for printing output");
	OptionParser::addOption("n", "Filename for printing output");




	//Parametros 

	OptionParser::addOption("ki", "");
	OptionParser::addOption("ko", "");
	OptionParser::addOption("vmod", "");
	OptionParser::addOption("gna", "");
	OptionParser::addOption("gcl", "");
	OptionParser::addOption("atp", "");
	OptionParser::addOption("use_gpu", "");

	OptionParser::parseOptions(argc, argv);



	float K_o_default = 5.40e+00;
	float g_CaL_default = 1.750e-04;
	float g_Na_default = 1.48380e+01;
	float K_i_default = 138.3; 
	float atp_default = 5.4E0;

	int method_index = OptionParser::foundOption("method") ? OptionParser::parsefloat("method") : 1;
	int model_index = 0;

	float dt = OptionParser::foundOption("dt") ? OptionParser::parsefloat("dt") : 0.1;
	float dt_save = OptionParser::foundOption("dt_save") ? OptionParser::parsefloat("dt_save") : 1;
	float tf = OptionParser::foundOption("tf") ? OptionParser::parsefloat("tf") : 400;
	int N = OptionParser::foundOption("n") ? OptionParser::parseInt("n") : 10000;

	bool use_gpu = OptionParser::foundOption("use_gpu") ? (OptionParser::parseInt("use_gpu")) == 1 ? true : false :  true;

	int np = int(tf / dt_save + 1);

	float* paramS = (float*)malloc(sizeof(float) * 5 * N);
	float* out;
	out = (float*)malloc(sizeof(float) *  np* N);



	ifstream myfile;
	myfile.open("m.txt");


	for (int i = 0; i < N; i++) {
		for (int j = 0; j < 5; j++) {
			myfile >> paramS[i + j * N];

		}
	}


	printf(" \n Problem: %d cells \n",N);
	
	if (use_gpu == true) {
		
		printf(" \n Solve by gpu Grid: %dx%d threads \n", N/10,10);
		float* param_g;
		gpuErrchk(cudaMalloc((void**)&param_g, sizeof(float)*4 * 5 * N));
		printf("aa");
		gpuErrchk(cudaMemcpy(param_g, paramS, sizeof(float) * 50 * N, cudaMemcpyHostToDevice));
		
		int nt = 100;
		int bx=10,by=N/(bx*nt);
		dim3 k(bx,by);
		float* out_g;
		int np = int(tf / dt_save + 1);
		gpuErrchk(cudaMalloc((void**)&out_g, sizeof(float) *np  * N));
		solveFixed << < k,nt>> > (out_g, dt, dt_save, tf, param_g);
		gpuErrchk(cudaPeekAtLastError());

		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaMemcpy(out, out_g, sizeof(float) * np * N, cudaMemcpyDeviceToHost));


		free(out_g);
		free(param_g);
	}

	else {
		printf(" \n Solve by cpu 4 threads \n");
#pragma omp parallel for  num_threads(4)x
		for (int i = 0; i < N; i++)
			solveFixedCpu(out, dt, dt_save, tf, paramS, i, N);

		free(paramS);
	}











	fstream output;
	output.open("out.txt", fstream::out);


	for (int i = 0; i < N; i++) {
		for (int j = 0; j < tf / dt_save; j++) {
			if (j != 0)
				output << " ";
			output << out[j + i * int(tf / dt_save)];
		}
		output << std::endl;
	}



	output.close();


	printf("\n OUTPUT FILE READY!\n");

	free(out); 
	return 0;
}


__global__ void solveFixed(float* out_g,float dt, float dt_save, float tf, float* args)
{
	
	unsigned int bid = blockIdx.y * gridDim.x +blockIdx.x ;
	unsigned int tid = threadIdx.x + (blockDim.x)*bid;
	printf("\n%d %d\n", tid, bid);


	float* Y_old_ = new float[18];
	int np = int(tf / dt_save);

	float* Y_new_ = new float[18];


	float** Tr = NULL;
	if (nStates_MKM_max > 0) {
		Tr = new float* [nStates_MKM_max];
		for (int i = 0; i < nStates_MKM_max; ++i)
			Tr[i] = new float[nStates_MKM_max];
	}
	// rhs will store the righ-hand-side values of NL and MK ODEs, and the coefficients a and b of the HH equations (for the RL method)
	float* rhs = new float[nStates + nStates_HH];
	float* algs = new float[nAlgs];
	
	
	unsigned int N = blockDim.x * gridDim.x * gridDim.y;
	initModel(args, Y_old_, args);

	//	cout << "[ ";
	float aux2 = Y_old_[0];
	float t_save = 0, aux = 0;
	int k = 0;
	for (float t = 0; t <= tf; t += dt) {



		step(Y_new_, args, algs, rhs, Y_old_, t, dt, Tr,tid,N);

		float dv = Y_new_[0] - Y_old_[0];
		if (dv * dv > aux * aux)
			aux = dv;
		for (int l = 0; l < nStates; l++) Y_old_[l] = Y_new_[l];

		t_save += dt;
		if (t_save >= dt_save) {

			out_g[k + tid * np] = Y_new_[0];
			k++;
			t_save = 0;
		//	std:printf("\n%d %d\n",tid, k + tid * np);
			printf("\n%d %f \n",k, Y_new_[0]);
			//	printf("\n%d\n", tid);
			
		}
	
	}
		//out_g[(tid + 1) * np - 1] = aux;

}

void solveFixedCpu(float* out_g, float dt, float dt_save, float tf, float* args, int tid, int N)
{

	float* Y_old_ = new float[18];
	int np = int(tf / dt_save);

	float* Y_new_ = new float[18];


	float** Tr = NULL;
	if (nStates_MKM_max > 0) {
		Tr = new float* [nStates_MKM_max];
		for (int i = 0; i < nStates_MKM_max; ++i)
			Tr[i] = new float[nStates_MKM_max];
	}
	// rhs will store the righ-hand-side values of NL and MK ODEs, and the coefficients a and b of the HH equations (for the RL method)
	float* rhs = new float[nStates + nStates_HH];
	float* algs = new float[nAlgs];
	float* params = new float[48];
	
	initModelC(params, Y_old_, args, tid, N);

	//	cout << "[ ";
	float aux2 = Y_old_[0];
	float t_save = 0, aux = 0;
	int k = 0;
	#pragma unroll
	for (float t = 0; t <= tf; t += dt) {



		stepC(Y_new_,args, algs, rhs, Y_old_, t, dt, Tr,tid,N);

		float dv = Y_new_[0] - Y_old_[0];
		if (dv * dv > aux * aux)
			aux = dv;
		for (int l = 0; l < nStates; l++) Y_old_[l] = Y_new_[l];

		t_save += dt;
		if (t_save >= dt_save) {

			out_g[k + tid * np] = Y_new_[0];
			k++;
			t_save = 0;



			;
		}
		out_g[(tid + 1) * np - 1] = aux;
	}
	//	cout << " -100 " << aux << "]";
}




ArgvParser::ArgvParser()
	: max_key(1),
	help_option(0) // must be smaller than max_key initially

{
	// nothing
}

ArgvParser::~ArgvParser()
{
	// nothing
}

void ArgvParser::reset()
{
	max_key = 1;
	option2key.clear();
	option2attribute.clear();
	option2descr.clear();
	option2value.clear();
	errorcode2descr.clear();
	argument_container.clear();
	intro_description.clear();
	error_option.clear();
	help_option = 0;
}

int ArgvParser::optionKey(const string& _name) const
{
	String2KeyMap::const_iterator it = option2key.find(_name);

	// if not found
	if (it == option2key.end())
		return(-1);

	return(it->second);
}

bool ArgvParser::isDefinedOption(const string& _name) const
{
	return(option2key.find(_name) != option2key.end());
}

bool ArgvParser::foundOption(const string& _name) const
{
	int key = optionKey(_name);

	// not defined -> cannot by found
	if (key == -1)
		return(false);

	// return whether the key of the given option name is in the hash of the
	// parsed options.
	return(option2value.find(key) != option2value.end());
}

string ArgvParser::optionValue(const string& _option) const
{
	int key = optionKey(_option);

	// not defined -> cannot by found
	if (key == -1)
	{
		cerr << "ArgvParser::optionValue(): Requested value of an option the parser did not find or does not know." << endl;
		return("");
	}

	return(option2value.find(key)->second);
}

ArgvParser::ParserResults
ArgvParser::parse(int _argc, char** _argv)
{
	bool finished_options = false; // flag whether an argument was found (options are passed)

	// loop over all command line arguments
	int i = 1; // argument counter
	while (i < _argc)
	{
		string argument = _argv[i];
		unsigned int key = 0;
		string option; // option name
		string value;  // option value

		// if argument is an option
		if (!isValidOptionString(argument))
		{
			// string is a real argument since values are processed elsewhere
			finished_options = true;
			argument_container.push_back(argument);
		}
		else // can be a long or multiple short options at this point
		{
			// check whether we already found an argument
			if (finished_options)
			{
				error_option = argument;
				return(ParserOptionAfterArgument); // return error code
			}
			// check for long options
			if (isValidLongOptionString(argument))
			{
				// handle long options

				// remove trailing '--'
				argument = argument.substr(2);
				// check for option value assignment 'option=value'
				splitOptionAndValue(argument, option, value);

				if (!isDefinedOption(option)) // is this a known option
				{
					error_option = option; // store the option that caused the error
					return(ParserUnknownOption); // return error code if not
				}

				// get the key of this option - now that we know that it is defined
				key = option2key.find(option)->second;
				if (key == help_option) // if help is requested return error code
					return(ParserHelpRequested);

				// do we need to extract a value
				// AND a value is not already assigned from the previous step
				if ((option2attribute.find(key)->second & OptionRequiresValue) && value.empty())
				{
					if (i + 1 >= _argc) // are there arguments left?
					{
						error_option = option; // store the option that caused the error
						return(ParserMissingValue); // the was no argument left although we need a value
					}

					string temp = _argv[i + 1]; // get the next element
					++i; // increase counter now that we moved forward

					if (isValidOptionString(temp))
					{
						error_option = option; // store the option that caused the error
						return(ParserMissingValue);  // missing option value
					}
					value = temp; // assign value
				}
				// add option-value map entry
				option2value[key] = value;
			}
			else // handle short options
			{
				argument = argument.substr(1);   // remove trailing '-'

				// check for option value assignment 'option=value'
				if (splitOptionAndValue(argument, option, value))
				{
					// there was an option <- value assignment
					if (option.length() > 1)
					{
						error_option = option; // store the option that caused the error
						return(ParserMalformedMultipleShortOption); // return error code if option has more than one character
					}

					if (!isDefinedOption(option)) // is this a known option
					{
						error_option = option; // store the option that caused the error
						return(ParserUnknownOption); // return error code if not
					}
					key = option2key.find(option)->second; // get the key for the extracted option name

					if (key == help_option) // if help is requested return error code
						return(ParserHelpRequested);

					// if value is still empty for some reason: we have an error
					if ((option2attribute.find(key)->second & OptionRequiresValue) && value.empty())
					{
						error_option = option; // store the option that caused the error
						return(ParserMissingValue);   // missing option value
					}
					else
						// add option-value map entry
						option2value[key] = value;
				}
				else // no '=' assignment: can be either multiple short options or
					// something like '-s 4'
				{
					// handle short options with value like '-s 4'
					option.clear();
					value.clear();

					if (argument.length() == 1) // if a single short option
					{
						if (!isDefinedOption(argument)) // is this a known option
						{
							error_option = argument; // store the option that caused the error
							return(ParserUnknownOption); // return error code if not
						}
						key = option2key.find(argument)->second; // get the key for the extracted option name

						if (key == help_option) // if help is requested return error code
							return(ParserHelpRequested);

						// check if option needs a value and next arg is not an option
						if ((option2attribute.find(key)->second & OptionRequiresValue))
						{
							if (i + 1 >= _argc) // are there arguments left?
							{
								error_option = argument; // store the option that caused the error
								return(ParserMissingValue); // the was no argument left although we need a value
							}
							string temp = _argv[i + 1]; // get the next element
							++i; // increase counter now that we moved forward

							if (isValidOptionString(temp))
							{
								error_option = argument; // store the option that caused the error
								return(ParserMissingValue);  // missing option value
							}
							// add option-value map entry
							option2value[key] = temp;

						}
						else // no value needed
						{
							option2value[key] = ""; // assign value
						}
					}
					else // handle multiple short option like '-svxgh'
					{
						unsigned int short_option_counter = 0; // position in the multiple short option string
						while (short_option_counter < argument.length()) // parse the whole string
						{
							option = argument[short_option_counter]; // get the option character

							if (!isDefinedOption(option)) // is this a known option
							{
								error_option = option; // store the option that caused the error
								return(ParserUnknownOption); // return error code if not
							}
							key = option2key.find(option)->second; // get the key for the extracted option name

							if (key == help_option) // if help is requested return error code
								return(ParserHelpRequested);

							option2value[key] = value;

							++short_option_counter; // advance one character forward
						}
					}
				}
			}
		}
		++i;
	}

	map<unsigned int, OptionAttributes>::iterator it;
	for (it = option2attribute.begin(); it != option2attribute.end(); it++)
	{
		// if the current option is required look if we got it
		if (it->second & OptionRequired)
		{
			// is the object missing
			if (option2value.find(it->first) == option2value.end())
			{
				// get the list of alternative names for this option
				list<string> alternatives = getAllOptionAlternatives(it->first);

				unsigned int count = 0;
				for (list<string>::const_iterator alt = alternatives.begin();
					alt != alternatives.end();
					++alt)
				{
					++count;
					// additional '-' for long options
					if (alt->length() > 1)
						error_option += "-";

					error_option += "-" + *alt;

					// alternatives to come?
					if (count < alternatives.size())
						error_option += ", "; // add separator
				}
				return(ParserRequiredOptionMissing);
			}
		}
	}

	return(NoParserError); // everthing went fine -> sucess
}

unsigned int ArgvParser::arguments() const
{
	return(argument_container.size());
}

string ArgvParser::argument(unsigned int _id) const
{
	if (_id >= arguments())
	{
		cerr << "ArgvParser::argument(): Request for non-existing argument." << endl;
		return ("");
	}
	else
		return(argument_container[_id]);
}

const vector<string>& ArgvParser::allArguments() const
{
	return(argument_container);
}

const string& ArgvParser::errorOption() const
{
	return(error_option);
}

std::string ArgvParser::parseErrorDescription(ParserResults _error_code) const
{
	string descr;

	switch (_error_code)
	{
	case ArgvParser::NoParserError:
		// no error -> nothing to do
		break;
	case ArgvParser::ParserUnknownOption:
		descr = "Unknown option: '" + errorOption() + "'";
		break;
	case ArgvParser::ParserMissingValue:
		descr = "Missing required value for option: '" + errorOption() + "'";
		break;
	case ArgvParser::ParserOptionAfterArgument:
		descr = "Misplaced option '" + errorOption() + "' detected. All option have to be BEFORE the first argument";
		break;
	case ArgvParser::ParserMalformedMultipleShortOption:
		descr = "Malformed short-options: '" + errorOption() + "'";
		break;
	case ArgvParser::ArgvParser::ParserRequiredOptionMissing:
		descr = "Required option missing: '" + errorOption() + "'";
		break;
	case ArgvParser::ParserHelpRequested: // help

		break;
	default:
		cerr << "ArgvParser::documentParserErrors(): Unknown error code" << endl;
	}

	return(descr);
}

bool ArgvParser::defineOption(const string& _name,
	const string& _descr,
	OptionAttributes _attrs)
{
	// do nothing if there already is an option of this name
	if (isDefinedOption(_name))
	{
		cerr << "ArgvParser::defineOption(): The option label equals an already defined option." << endl;
		return(false);
	}

	// no digits as short options allowed
	if (_name.length() == 1 && isDigit(_name[0]))
	{
		cerr << "ArgvParser::defineOption(): Digits as short option labels are not allowd." << endl;
		return(false);
	}

	option2key[_name] = max_key;     // give the option a unique key

	// store the option attributes
	option2attribute[max_key] = _attrs;

	// store the option description if there is one
	if (_descr.length())
		option2descr[max_key] = _descr;

	// inc the key counter
	++max_key;

	return(true);
}

bool ArgvParser::defineOptionAlternative(const string& _original,
	const string& _alternative)
{
	// do nothing if there already is no option of this name
	if (!isDefinedOption(_original))
	{
		cerr << "ArgvParser::defineOptionAlternative(): Original option label is not a defined option." << endl;
		return(false);
	}

	// AND no digits as short options allowed
	if (_alternative.length() == 1 && isDigit(_alternative[0]))
	{
		cerr << "ArgvParser::defineOptionAlternative(): Digits as short option labels are not allowd." << endl;
		return(false);
	}

	// AND do nothing if there already is an option with the alternativ name
	if (isDefinedOption(_alternative))
	{
		cerr << "ArgvParser::defineOptionAlternative(): The alternative option label equals an already defined option." << endl;
		return(false);
	}

	option2key[_alternative] = optionKey(_original);

	return(true);
}


bool ArgvParser::setHelpOption(const string& _shortname,
	const string& _longname,
	const string& _descr)
{
	// do nothing if any name is already in use
	if (isDefinedOption(_shortname) || isDefinedOption(_longname))
	{
		cerr << "ArgvParser::setHelpOption(): Short or long help option label equals an already defined option." << endl;
		return(false);
	}

	// define the help option's short name and the alternative
	// longname
	defineOption(_shortname, _descr, NoOptionAttribute);
	defineOptionAlternative(_shortname, _longname);

	help_option = max_key - 1; // store the key in a special member

	return(true);
}

void ArgvParser::addErrorCode(int _code, const string& _descr)
{
	errorcode2descr[_code] = _descr;
}

void ArgvParser::setIntroductoryDescription(const string& _descr)
{
	intro_description = _descr;
}

list<string> ArgvParser::getAllOptionAlternatives(unsigned int _key) const
{
	// keys go here
	list<string> keys;
	// for all container elements
	for (map<string, unsigned int>::const_iterator it = option2key.begin();
		it != option2key.end();
		++it)
	{
		if (it->second == _key)
			keys.push_back(it->first);
	}
	return(keys);
}

bool CommandLineProcessing::isDigit(const char& _char)
{
	if (_char == '0' || _char == '1' || _char == '2' || _char == '3'
		|| _char == '4' || _char == '5' || _char == '6' || _char == '7'
		|| _char == '8' || _char == '9')
		return(true);
	else
		return(false);
}

bool CommandLineProcessing::isValidOptionString(const string& _string)
{
	// minimal short option length is 2
	if (_string.length() < 2)
		return(false);

	// is it an option (check for '-' as first character)
	if (_string.compare(0, 1, "-"))
		return(false);

	// not an option if just '--'
	if (_string.length() == 2 && _string == "--")
		return(false);

	// it might still be a negative number
	// (but not if there is no digit afterwards)
	if (isDigit(_string[1]))
		return(false);

	// let's consider this an option
	return(true);
}

bool CommandLineProcessing::isValidLongOptionString(const string& _string)
{
	if (_string.length() < 4) // must be at least '--??'
		return(false);

	// is it an option (check for '--')
	if (_string.compare(0, 2, "--"))
		return(false);
	else
		return(true);
}

bool CommandLineProcessing::splitOptionAndValue(const string& _string,
	string& _option, string& _value)
{
	// string token container
	std::vector<string> tokens;

	// split string by '=' delimiter
	splitString(tokens, _string, "=");

	// check for option value assignment 'option=value'
	if (tokens.size() < 2)
	{
		_option = _string; // the option is the whole string
		return(false);
	}

	// separate option and value
	_option = tokens[0];

	// concat all remaining tokens to the value string
	for (unsigned int i = 1; i < tokens.size(); ++i)
	{
		_value.append(tokens[i]);
	}

	return(true);
}

string CommandLineProcessing::trimmedString(const std::string& _str)
{
	// no string no work
	if (_str.length() == 0)
		return _str;

	string::size_type start_pos = _str.find_first_not_of(" \a\b\f\n\r\t\v");
	string::size_type end_pos = _str.find_last_not_of(" \a\b\f\n\r\t\v");

	// check whether there was any non-whitespace
	if (start_pos == string::npos)
		return("");

	return string(_str, start_pos, end_pos - start_pos + 1);
}

bool CommandLineProcessing::expandRangeStringToUInt(const std::string& _string,
	std::vector< unsigned int >& _expanded)
{
	list<string> tokens;
	// split string by delimiter
	splitString(tokens, _string, ",");

	// loop over all entries
	for (list<string>::const_iterator it = tokens.begin(); it != tokens.end(); it++)
	{
		const string& entry = *it; // convenience reference

#ifdef ARGVPARSER_DEBUG

		cout << "TOKEN: " << entry << endl;
#endif

		// if range was given
		if (entry.find("-") != string::npos)
		{
			// split into upper and lower border
			list<string> range_borders;
			splitString(range_borders, entry, "-");

			// fail if insane range spec
			if (range_borders.size() != 2)
				return(false);

			int first = atoi(range_borders.begin()->c_str());
			int second = atoi((++range_borders.begin())->c_str());

			// write id in increasing order
			if (first <= second)

			{
				for (int j = first; j <= second; ++j)
				{
					_expanded.push_back(j);
				}
			}
			else // write id in decreasing order
			{
				for (int k = first; k >= second; k--)
				{
					_expanded.push_back(k);
				}
			}
		}
		else // single number was given
			_expanded.push_back(atoi(entry.c_str())); // store id
	}

	return(true);
}

std::string CommandLineProcessing::formatString(const std::string& _string,
	unsigned int _width,
	unsigned int _indent)
{
	// if insane parameters do nothing
	if (_indent >= _width)
		return(_string);

	// list of lines of the formated string
	list<string> lines;

	// current position in the string
	unsigned int pos = 0;

	// till the end of the string
	while (pos < _string.length())
	{
		// get the next line of the string
		string line = _string.substr(pos, _width - _indent);

#ifdef ARGVPARSER_DEBUG

		cout << "EXTRACT: '" << line << "'" << endl;
#endif

		// check for newlines in the line and break line at first occurence (if any)
		string::size_type first_newline = line.find_first_of("\n");
		if (first_newline != string::npos)
		{
			line = line.substr(0, first_newline);
		}

		// we need to check for possible breaks within words only if the extracted
		// line spans the whole allowed width
		bool check_truncation = true;
		if (line.length() < _width - _indent)
			check_truncation = false;

		// remove unecessary whitespace at front and back
		line = trimmedString(line);

#ifdef ARGVPARSER_DEBUG

		cout << "TRIMMED: '" << line << "'" << endl;
#endif

		// only perform truncation if there was enough data for a full line
		if (!check_truncation)
			pos += line.length() + 1;
		else
		{
			// look for the last whitespace character
			string::size_type last_white_space = line.find_last_of(" \a\b\f\n\r\t\v");

			if (last_white_space != string::npos) // whitespace found!
			{
				// truncated the line at the last whitespace
				line = string(line, 0, last_white_space);
				pos += last_white_space + 1;
			}
			else // no whitespace found
				// rude break! we can leave the line in its current state
				pos += _width - _indent;
		}

		if (!line.empty())
		{
#ifdef ARGVPARSER_DEBUG
			cout << "UNINDEN: '" << line << "'" << endl;
#endif

			if (_indent)
				line.insert(0, _indent, ' ');

#ifdef ARGVPARSER_DEBUG

			cout << "INDENT: '" << line << "'" << endl;
#endif

			lines.push_back(line);
		}
	}

	// concat the formated string
	string formated;
	bool first = true;
	// for all lines
	for (list<string>::iterator it = lines.begin(); it != lines.end(); ++it)
	{
		// prefix with newline if not first
		if (!first)
			formated += "\n";
		else
			first = false;

		formated += *it;
	}
	return(formated);
}

ArgvParser OptionParser::cmd;

void OptionParser::addOption(const string& option, const string& desc) {
	cmd.defineOption(option, desc, ArgvParser::NoOptionAttribute);
}

void OptionParser::setup() {
	//define error codes
	cmd.addErrorCode(0, "Success");
	cmd.addErrorCode(1, "Error");
}

void OptionParser::parseOptions(int argc, char** argv) {
	int result = cmd.parse(argc, argv);

	if (result != ArgvParser::NoParserError) {
		cout << "Parse ERROR!!!: " << cmd.parseErrorDescription(result) << endl;
		exit(1);
	}
}

string OptionParser::optionValue(const string& option) {
	return cmd.optionValue(option);
}

bool OptionParser::foundOption(const string& option) {
	return cmd.foundOption(option);
}

OptionParser::OptionParser() {
	this->myCmd.addErrorCode(0, "Success");
	this->myCmd.addErrorCode(1, "Error");
}

void OptionParser::add(const string& option, const string& desc) {
	myCmd.defineOption(option, desc, ArgvParser::NoOptionAttribute);
}

int OptionParser::parse(string optionsText) {
	vector<char*> args;
	istringstream iss(optionsText);
	string token;
	int optionsCount = 0;
	while (iss >> token) {
		char* arg = new char[token.size() + 1];
		copy(token.begin(), token.end(), arg);
		arg[token.size()] = '\0';
		args.push_back(arg);
		optionsCount++;
	}
	args.push_back(0);
	int result = myCmd.parse(optionsCount, &args[0]);

	if (result != ArgvParser::NoParserError) {
		cout << "Parse ERROR!!!: " << myCmd.parseErrorDescription(result) << endl;
		return 0;
	}
	return 1;
}
string OptionParser::value(const string& option) {
	return myCmd.optionValue(option);
}
bool OptionParser::has(const string& option) {
	return myCmd.foundOption(option);
}

float OptionParser::parsefloat(const string& option) {
	float aux;
	sscanf(cmd.optionValue(option).c_str(), "%lf", &aux);
	return aux;
}

int OptionParser::parseInt(const string& option) {
	int aux;
	sscanf(cmd.optionValue(option).c_str(), "%d", &aux);
	return aux;
}
